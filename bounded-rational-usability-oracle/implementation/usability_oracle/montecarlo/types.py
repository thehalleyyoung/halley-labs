"""
usability_oracle.montecarlo.types — Data types for the Monte Carlo trajectory engine.

Provides immutable value types for configuring and collecting results from
Monte Carlo sampling of user trajectories through the usability MDP.
Supports importance sampling, stratified sampling, and variance-reduction
diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, NewType, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# NewType aliases
# ---------------------------------------------------------------------------

SampleId = NewType("SampleId", int)
"""Unique identifier for a single Monte Carlo sample."""

SeedValue = NewType("SeedValue", int)
"""Random seed for reproducible sampling."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

@unique
class SamplingStrategy(Enum):
    """Monte Carlo sampling strategy."""

    DIRECT = "direct"
    """Sample trajectories directly from the policy π(a|s)."""

    IMPORTANCE = "importance"
    """Importance sampling with a proposal distribution q(a|s)."""

    STRATIFIED = "stratified"
    """Stratified sampling over initial states for variance reduction."""

    ANTITHETIC = "antithetic"
    """Antithetic variates: pair each sample with its complement."""

    QUASI_MONTE_CARLO = "quasi_monte_carlo"
    """Low-discrepancy (Sobol/Halton) sequence for uniform coverage."""


@unique
class TerminationReason(Enum):
    """Why a trajectory simulation terminated."""

    GOAL_REACHED = "goal_reached"
    """Agent arrived at a goal state."""

    MAX_STEPS = "max_steps"
    """Step budget exhausted."""

    CYCLE_DETECTED = "cycle_detected"
    """Agent revisited a state, forming a cycle."""

    DEAD_END = "dead_end"
    """No available actions from the current state."""


# ═══════════════════════════════════════════════════════════════════════════
# MCConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MCConfig:
    """Configuration for Monte Carlo trajectory sampling.

    Attributes:
        num_samples: Total number of trajectories to sample.
        max_trajectory_length: Maximum steps per trajectory before
            forced termination.
        strategy: Sampling strategy (direct, importance, stratified, …).
        seed: Random seed for reproducibility.  If ``None`` a fresh
            entropy source is used.
        burn_in: Number of initial samples to discard (for MCMC-based
            strategies).
        thin_interval: Keep every *k*-th sample to reduce autocorrelation.
        parallel_workers: Number of parallel workers for trajectory
            generation.  ``1`` means sequential execution.
        detect_cycles: If ``True``, terminate trajectories that revisit
            a state.
        confidence_level: Confidence level for interval estimates
            (e.g. 0.95).
    """

    num_samples: int = 10_000
    max_trajectory_length: int = 200
    strategy: SamplingStrategy = SamplingStrategy.DIRECT
    seed: Optional[int] = None
    burn_in: int = 0
    thin_interval: int = 1
    parallel_workers: int = 1
    detect_cycles: bool = True
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be ≥ 1, got {self.num_samples}")
        if self.max_trajectory_length < 1:
            raise ValueError("max_trajectory_length must be ≥ 1")
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be ≥ 1")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be in (0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "max_trajectory_length": self.max_trajectory_length,
            "strategy": self.strategy.value,
            "seed": self.seed,
            "burn_in": self.burn_in,
            "thin_interval": self.thin_interval,
            "parallel_workers": self.parallel_workers,
            "detect_cycles": self.detect_cycles,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MCConfig:
        return cls(
            num_samples=int(d["num_samples"]),
            max_trajectory_length=int(d["max_trajectory_length"]),
            strategy=SamplingStrategy(d.get("strategy", "direct")),
            seed=d.get("seed"),
            burn_in=int(d.get("burn_in", 0)),
            thin_interval=int(d.get("thin_interval", 1)),
            parallel_workers=int(d.get("parallel_workers", 1)),
            detect_cycles=bool(d.get("detect_cycles", True)),
            confidence_level=float(d.get("confidence_level", 0.95)),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ImportanceWeight
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ImportanceWeight:
    """Importance-sampling weight for a single trajectory.

    The weight is  w(τ) = p_π(τ) / q(τ)  where p_π is the target
    distribution and q is the proposal.  Self-normalised weights are
    w̃ᵢ = wᵢ / Σⱼ wⱼ.

    Attributes:
        sample_id: Identifier of the trajectory this weight belongs to.
        raw_weight: Un-normalised importance weight  w(τ).
        log_weight: log w(τ) for numerical stability.
        normalised_weight: Self-normalised weight w̃.
    """

    sample_id: int
    raw_weight: float
    log_weight: float
    normalised_weight: float

    @property
    def effective(self) -> bool:
        """Whether this sample carries meaningful weight (> 1e-12)."""
        return self.normalised_weight > 1e-12

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "raw_weight": self.raw_weight,
            "log_weight": self.log_weight,
            "normalised_weight": self.normalised_weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ImportanceWeight:
        return cls(
            sample_id=int(d["sample_id"]),
            raw_weight=float(d["raw_weight"]),
            log_weight=float(d["log_weight"]),
            normalised_weight=float(d["normalised_weight"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# VarianceEstimate
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class VarianceEstimate:
    """Variance estimate for a Monte Carlo estimator.

    Reports both the raw sample variance and diagnostics that indicate
    whether variance reduction is necessary.

    Attributes:
        sample_variance: Var[X̂] — variance of the estimator.
        standard_error: SE = √(Var / n).
        coefficient_of_variation: CV = σ / μ.
        effective_sample_size: ESS for importance sampling;
            ESS = (Σwᵢ)² / Σwᵢ².  For direct sampling ESS = n.
        ess_ratio: ESS / n — fraction of samples that are "effective".
            Below 0.1 suggests severe weight degeneracy.
    """

    sample_variance: float
    standard_error: float
    coefficient_of_variation: float
    effective_sample_size: float
    ess_ratio: float

    @property
    def is_degenerate(self) -> bool:
        """Whether the importance weights are severely degenerate."""
        return self.ess_ratio < 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_variance": self.sample_variance,
            "standard_error": self.standard_error,
            "coefficient_of_variation": self.coefficient_of_variation,
            "effective_sample_size": self.effective_sample_size,
            "ess_ratio": self.ess_ratio,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VarianceEstimate:
        return cls(
            sample_variance=float(d["sample_variance"]),
            standard_error=float(d["standard_error"]),
            coefficient_of_variation=float(d["coefficient_of_variation"]),
            effective_sample_size=float(d["effective_sample_size"]),
            ess_ratio=float(d["ess_ratio"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SampleStatistics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SampleStatistics:
    """Aggregate statistics computed from a set of trajectory samples.

    Attributes:
        mean_cost: Sample mean of total trajectory costs.
        median_cost: Sample median of total trajectory costs.
        std_cost: Sample standard deviation.
        min_cost: Minimum observed trajectory cost.
        max_cost: Maximum observed trajectory cost.
        percentiles: Mapping of percentile → cost value (e.g. {5: 2.1, 95: 8.7}).
        mean_length: Average trajectory length (steps).
        goal_reach_rate: Fraction of trajectories reaching a goal state.
        variance_estimate: Detailed variance diagnostics.
    """

    mean_cost: float
    median_cost: float
    std_cost: float
    min_cost: float
    max_cost: float
    percentiles: Dict[int, float]
    mean_length: float
    goal_reach_rate: float
    variance_estimate: VarianceEstimate

    @property
    def ci_lower(self) -> float:
        """Lower bound of the 95 % confidence interval for the mean cost."""
        return self.mean_cost - 1.96 * self.variance_estimate.standard_error

    @property
    def ci_upper(self) -> float:
        """Upper bound of the 95 % confidence interval for the mean cost."""
        return self.mean_cost + 1.96 * self.variance_estimate.standard_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_cost": self.mean_cost,
            "median_cost": self.median_cost,
            "std_cost": self.std_cost,
            "min_cost": self.min_cost,
            "max_cost": self.max_cost,
            "percentiles": self.percentiles,
            "mean_length": self.mean_length,
            "goal_reach_rate": self.goal_reach_rate,
            "variance_estimate": self.variance_estimate.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SampleStatistics:
        return cls(
            mean_cost=float(d["mean_cost"]),
            median_cost=float(d["median_cost"]),
            std_cost=float(d["std_cost"]),
            min_cost=float(d["min_cost"]),
            max_cost=float(d["max_cost"]),
            percentiles={int(k): float(v) for k, v in d["percentiles"].items()},
            mean_length=float(d["mean_length"]),
            goal_reach_rate=float(d["goal_reach_rate"]),
            variance_estimate=VarianceEstimate.from_dict(d["variance_estimate"]),
        )


# ═══════════════════════════════════════════════════════════════════════════
# TrajectoryBundle
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class TrajectoryBundle:
    """A bundle of sampled trajectories with aggregate statistics.

    This is the primary output of the Monte Carlo trajectory engine.
    Individual trajectories are referenced by index; the bundle stores
    their costs, lengths, termination reasons, and (optionally) importance
    weights.

    Attributes:
        num_trajectories: Number of trajectories in the bundle.
        costs: Total cost of each trajectory (length = num_trajectories).
        lengths: Step count of each trajectory.
        termination_reasons: Why each trajectory ended.
        importance_weights: Per-trajectory importance weights (``None``
            for direct sampling).
        statistics: Aggregate sample statistics.
        config: Configuration used for sampling.
        wall_clock_seconds: Total wall-clock time for sampling.
    """

    num_trajectories: int
    costs: Tuple[float, ...]
    lengths: Tuple[int, ...]
    termination_reasons: Tuple[TerminationReason, ...]
    importance_weights: Optional[Tuple[ImportanceWeight, ...]]
    statistics: SampleStatistics
    config: MCConfig
    wall_clock_seconds: float

    def __post_init__(self) -> None:
        n = self.num_trajectories
        if len(self.costs) != n:
            raise ValueError("costs length must match num_trajectories")
        if len(self.lengths) != n:
            raise ValueError("lengths length must match num_trajectories")
        if len(self.termination_reasons) != n:
            raise ValueError("termination_reasons length must match num_trajectories")

    @property
    def goal_reached_count(self) -> int:
        """Number of trajectories that reached a goal state."""
        return sum(
            1 for r in self.termination_reasons
            if r == TerminationReason.GOAL_REACHED
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "num_trajectories": self.num_trajectories,
            "costs": list(self.costs),
            "lengths": list(self.lengths),
            "termination_reasons": [r.value for r in self.termination_reasons],
            "statistics": self.statistics.to_dict(),
            "config": self.config.to_dict(),
            "wall_clock_seconds": self.wall_clock_seconds,
        }
        if self.importance_weights is not None:
            d["importance_weights"] = [w.to_dict() for w in self.importance_weights]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrajectoryBundle:
        iw_raw = d.get("importance_weights")
        return cls(
            num_trajectories=int(d["num_trajectories"]),
            costs=tuple(d["costs"]),
            lengths=tuple(d["lengths"]),
            termination_reasons=tuple(
                TerminationReason(r) for r in d["termination_reasons"]
            ),
            importance_weights=(
                tuple(ImportanceWeight.from_dict(w) for w in iw_raw)
                if iw_raw is not None
                else None
            ),
            statistics=SampleStatistics.from_dict(d["statistics"]),
            config=MCConfig.from_dict(d["config"]),
            wall_clock_seconds=float(d["wall_clock_seconds"]),
        )
