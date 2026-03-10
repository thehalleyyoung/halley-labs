"""
usability_oracle.montecarlo.statistics — Trajectory statistics computation.

Provides numerically stable computation of mean, variance, quantiles,
tail risk, bottleneck frequency, path entropy, and hitting times from
sampled trajectory bundles.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from usability_oracle.montecarlo.types import (
    ImportanceWeight,
    TerminationReason,
    TrajectoryBundle,
    VarianceEstimate,
)


# ═══════════════════════════════════════════════════════════════════════════
# TrajectoryStatistics
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryStatistics:
    """Compute statistics from trajectory bundles.

    All methods accept a :class:`TrajectoryBundle` or raw cost arrays.
    Numerically stable algorithms (Welford's) are used throughout.
    """

    # ------------------------------------------------------------------
    # Mean cost
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mean_cost(
        trajectories: TrajectoryBundle,
    ) -> float:
        """Compute mean total cost across trajectories.

        For importance-weighted samples:
            μ̂ = Σᵢ w̃ᵢ · Cᵢ

        For direct samples:
            μ̂ = (1/n) Σᵢ Cᵢ

        Uses Welford's algorithm for direct samples.

        Parameters:
            trajectories: Sampled trajectory bundle.

        Returns:
            Estimated mean cost.
        """
        costs = trajectories.costs
        n = len(costs)
        if n == 0:
            return 0.0

        if trajectories.importance_weights is not None:
            w = np.array(
                [iw.normalised_weight for iw in trajectories.importance_weights],
                dtype=np.float64,
            )
            return float(np.sum(w * np.array(costs)))

        # Welford's online mean
        mean = 0.0
        for i, c in enumerate(costs, 1):
            mean += (c - mean) / i
        return mean

    # ------------------------------------------------------------------
    # Cost variance
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cost_variance(
        trajectories: TrajectoryBundle,
    ) -> float:
        """Compute variance of trajectory costs with Bessel correction.

        Welford's online algorithm:
            δ  = Cᵢ − μ̂_{i−1}
            μ̂ᵢ = μ̂_{i−1} + δ/i
            M₂ᵢ = M₂_{i−1} + δ·(Cᵢ − μ̂ᵢ)
            σ̂² = M₂ₙ / (n−1)

        Parameters:
            trajectories: Sampled trajectory bundle.

        Returns:
            Sample variance (with Bessel correction).
        """
        costs = trajectories.costs
        n = len(costs)
        if n < 2:
            return 0.0

        if trajectories.importance_weights is not None:
            w = np.array(
                [iw.normalised_weight for iw in trajectories.importance_weights],
                dtype=np.float64,
            )
            arr = np.array(costs, dtype=np.float64)
            mean = float(np.sum(w * arr))
            return float(np.sum(w * (arr - mean) ** 2))

        # Welford's online algorithm
        mean = 0.0
        m2 = 0.0
        for i, x in enumerate(costs, 1):
            delta = x - mean
            mean += delta / i
            delta2 = x - mean
            m2 += delta * delta2

        return m2 / (n - 1)

    # ------------------------------------------------------------------
    # Cost quantiles
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cost_quantiles(
        trajectories: TrajectoryBundle,
        quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> Dict[float, float]:
        """Compute empirical quantiles of trajectory costs.

        Uses numpy's linear interpolation method for quantile estimation.

        Parameters:
            trajectories: Sampled trajectory bundle.
            quantiles: Sequence of quantile levels in [0, 1].

        Returns:
            Mapping quantile → cost value.
        """
        costs = trajectories.costs
        if not costs:
            return {q: 0.0 for q in quantiles}

        arr = np.array(costs, dtype=np.float64)
        result: Dict[float, float] = {}
        for q in quantiles:
            q_clamped = max(0.0, min(1.0, q))
            result[q] = float(np.percentile(arr, q_clamped * 100))
        return result

    # ------------------------------------------------------------------
    # Empirical CDF
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cost_cdf(
        trajectories: TrajectoryBundle,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute empirical CDF of trajectory costs.

        F̂(c) = (1/n) Σᵢ 𝟙{Cᵢ ≤ c}

        Parameters:
            trajectories: Sampled trajectory bundle.
            n_points: Number of evaluation points.

        Returns:
            ``(cost_values, cdf_values)`` arrays suitable for plotting.
        """
        costs = trajectories.costs
        if not costs:
            return np.array([]), np.array([])

        arr = np.sort(np.array(costs, dtype=np.float64))
        n = len(arr)
        cdf = np.arange(1, n + 1) / n

        if n <= n_points:
            return arr, cdf

        # Subsample for large datasets
        indices = np.linspace(0, n - 1, n_points, dtype=int)
        return arr[indices], cdf[indices]

    # ------------------------------------------------------------------
    # Tail risk (CVaR)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_tail_risk(
        trajectories: TrajectoryBundle,
        threshold: float,
    ) -> Tuple[float, float]:
        """Compute tail risk: P(cost > threshold) and CVaR.

        The Conditional Value at Risk (CVaR) at level α is:
            CVaR_α = E[C | C > VaR_α]

        Here we compute:
            P(C > threshold) — tail probability
            CVaR(threshold)  — E[C | C > threshold]

        Parameters:
            trajectories: Sampled trajectory bundle.
            threshold: Cost threshold.

        Returns:
            ``(tail_probability, conditional_mean)`` where
            conditional_mean is the CVaR.  Returns (0, 0) if no
            trajectories exceed the threshold.
        """
        costs = trajectories.costs
        if not costs:
            return 0.0, 0.0

        arr = np.array(costs, dtype=np.float64)
        exceedances = arr[arr > threshold]
        n = len(arr)

        tail_prob = len(exceedances) / n if n > 0 else 0.0
        cvar = float(np.mean(exceedances)) if len(exceedances) > 0 else 0.0

        return tail_prob, cvar

    # ------------------------------------------------------------------
    # Bottleneck frequency
    # ------------------------------------------------------------------

    @staticmethod
    def compute_bottleneck_frequency(
        trajectories: TrajectoryBundle,
        state_sequences: Optional[Sequence[Sequence[str]]] = None,
    ) -> Dict[str, float]:
        """Compute per-state visit frequency across trajectories.

        A "bottleneck" is a state visited disproportionately often.
        Frequency is normalised by the total number of trajectories.

        Parameters:
            trajectories: Sampled trajectory bundle (used for n).
            state_sequences: Per-trajectory state sequences.  If not
                provided, returns an empty dict.

        Returns:
            Mapping state → visit fraction (visits / n_trajectories).
        """
        if state_sequences is None or not state_sequences:
            return {}

        counter: Counter[str] = Counter()
        for seq in state_sequences:
            for s in seq:
                counter[s] += 1

        n = trajectories.num_trajectories
        if n == 0:
            return {}

        return {state: count / n for state, count in counter.most_common()}

    # ------------------------------------------------------------------
    # Path entropy
    # ------------------------------------------------------------------

    @staticmethod
    def compute_path_entropy(
        trajectories: TrajectoryBundle,
        state_sequences: Optional[Sequence[Tuple[str, ...]]] = None,
    ) -> float:
        """Compute Shannon entropy of the empirical path distribution.

        H(paths) = − Σ_τ p̂(τ) log p̂(τ)

        where p̂(τ) is the empirical frequency of each unique path.
        Higher entropy indicates more diverse user trajectories.

        Parameters:
            trajectories: Sampled trajectory bundle.
            state_sequences: Per-trajectory state tuples.  If not
                provided, uses trajectory lengths as a proxy
                (entropy of the length distribution).

        Returns:
            Shannon entropy in nats.
        """
        if state_sequences is not None and len(state_sequences) > 0:
            counter: Counter[Tuple[str, ...]] = Counter(state_sequences)
            n = sum(counter.values())
            if n == 0:
                return 0.0
            entropy = 0.0
            for count in counter.values():
                p = count / n
                if p > 0:
                    entropy -= p * math.log(p)
            return entropy

        # Fall back: entropy of the length distribution
        lengths = trajectories.lengths
        if not lengths:
            return 0.0
        counter_len: Counter[int] = Counter(lengths)
        n = sum(counter_len.values())
        entropy = 0.0
        for count in counter_len.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    # ------------------------------------------------------------------
    # Hitting time
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hitting_time(
        trajectories: TrajectoryBundle,
        target_states: Set[str],
        state_sequences: Optional[Sequence[Sequence[str]]] = None,
    ) -> Dict[str, float]:
        """Compute first-passage time statistics to target states.

        For each trajectory, the hitting time is the first step index
        at which a target state is reached.  Returns mean, median, std,
        and the fraction of trajectories that never hit any target.

        Parameters:
            trajectories: Sampled trajectory bundle.
            target_states: Set of target state identifiers.
            state_sequences: Per-trajectory state sequences.

        Returns:
            Dict with keys ``"mean"``, ``"median"``, ``"std"``,
            ``"miss_rate"`` (fraction that never hit a target).
        """
        if state_sequences is None or not state_sequences:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "miss_rate": 1.0}

        hitting_times: List[int] = []
        misses = 0

        for seq in state_sequences:
            found = False
            for step_idx, state in enumerate(seq):
                if state in target_states:
                    hitting_times.append(step_idx + 1)  # 1-indexed
                    found = True
                    break
            if not found:
                misses += 1

        n_total = len(state_sequences)
        miss_rate = misses / n_total if n_total > 0 else 1.0

        if not hitting_times:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "miss_rate": miss_rate}

        arr = np.array(hitting_times, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "miss_rate": miss_rate,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_mean_cost(trajectories: TrajectoryBundle) -> float:
    """Compute mean total cost across trajectories."""
    return TrajectoryStatistics.compute_mean_cost(trajectories)


def compute_cost_variance(trajectories: TrajectoryBundle) -> float:
    """Compute variance of trajectory costs with Bessel correction."""
    return TrajectoryStatistics.compute_cost_variance(trajectories)


def compute_cost_quantiles(
    trajectories: TrajectoryBundle,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> Dict[float, float]:
    """Compute empirical quantiles of trajectory costs."""
    return TrajectoryStatistics.compute_cost_quantiles(trajectories, quantiles)


def compute_cost_cdf(
    trajectories: TrajectoryBundle,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical CDF of trajectory costs."""
    return TrajectoryStatistics.compute_cost_cdf(trajectories, n_points)


def compute_tail_risk(
    trajectories: TrajectoryBundle,
    threshold: float,
) -> Tuple[float, float]:
    """Compute tail risk: P(cost > threshold) and CVaR."""
    return TrajectoryStatistics.compute_tail_risk(trajectories, threshold)


def compute_path_entropy(
    trajectories: TrajectoryBundle,
    state_sequences: Optional[Sequence[Tuple[str, ...]]] = None,
) -> float:
    """Compute Shannon entropy of the empirical path distribution."""
    return TrajectoryStatistics.compute_path_entropy(trajectories, state_sequences)
