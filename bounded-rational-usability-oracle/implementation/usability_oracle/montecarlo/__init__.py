"""
usability_oracle.montecarlo — Monte Carlo trajectory sampling engine.

Provides configurable trajectory sampling from the usability MDP, with
support for importance sampling, stratified sampling, and variance
reduction.  Re-exports all public types, protocols, and implementations::

    from usability_oracle.montecarlo import MCConfig, TrajectoryBundle
    from usability_oracle.montecarlo import TrajectorySamplerImpl, ParallelMCExecutor
"""

from __future__ import annotations

from usability_oracle.montecarlo.types import (
    ImportanceWeight,
    MCConfig,
    SampleStatistics,
    SamplingStrategy,
    TerminationReason,
    TrajectoryBundle,
    VarianceEstimate,
)

from usability_oracle.montecarlo.protocols import (
    ParallelExecutor,
    TrajectorySampler,
    VarianceReducer,
)

from usability_oracle.montecarlo.sampler import (
    TrajectorySamplerImpl,
    sample_batch,
    sample_trajectory,
)

from usability_oracle.montecarlo.variance_reduction import (
    AntitheticVariates,
    ControlVariates,
    ImportanceSampling,
    StratifiedSampling,
    compute_effective_sample_size,
    compute_optimal_proposal,
    rao_blackwellize,
)

from usability_oracle.montecarlo.parallel import (
    ParallelMCExecutor,
    adaptive_allocation,
    chunk_trajectories,
    merge_statistics,
)

from usability_oracle.montecarlo.statistics import (
    TrajectoryStatistics,
    compute_cost_cdf,
    compute_cost_quantiles,
    compute_cost_variance,
    compute_mean_cost,
    compute_path_entropy,
    compute_tail_risk,
)

from usability_oracle.montecarlo.diagnostics import MCDiagnostics

__all__ = [
    # types
    "ImportanceWeight",
    "MCConfig",
    "SampleStatistics",
    "SamplingStrategy",
    "TerminationReason",
    "TrajectoryBundle",
    "VarianceEstimate",
    # protocols
    "ParallelExecutor",
    "TrajectorySampler",
    "VarianceReducer",
    # implementations
    "TrajectorySamplerImpl",
    "ParallelMCExecutor",
    "MCDiagnostics",
    "TrajectoryStatistics",
    # variance reduction
    "ControlVariates",
    "AntitheticVariates",
    "StratifiedSampling",
    "ImportanceSampling",
    "compute_effective_sample_size",
    "compute_optimal_proposal",
    "rao_blackwellize",
    # parallel
    "adaptive_allocation",
    "chunk_trajectories",
    "merge_statistics",
    # statistics
    "compute_mean_cost",
    "compute_cost_variance",
    "compute_cost_quantiles",
    "compute_cost_cdf",
    "compute_tail_risk",
    "compute_path_entropy",
    # sampler convenience
    "sample_batch",
    "sample_trajectory",
]
