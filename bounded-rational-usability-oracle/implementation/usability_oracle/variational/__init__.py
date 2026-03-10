"""
usability_oracle.variational — Free-energy variational solver.

Implements bounded-rational policy computation via variational free-energy
minimisation:  F[π] = E_π[C(τ)] − (1/β) H[π].

Re-exports all public types, protocols, and implementations::

    from usability_oracle.variational import (
        VariationalConfig, FreeEnergyResult,
        FreeEnergyComputer, CapacityEstimatorImpl, VariationalOptimizer,
    )
"""

from __future__ import annotations

from usability_oracle.variational.types import (
    CapacityProfile,
    ConvergenceInfo,
    ConvergenceStatus,
    FreeEnergyResult,
    KLDivergenceResult,
    ObjectiveType,
    VariationalConfig,
)

from usability_oracle.variational.protocols import (
    CapacityEstimator,
    ObjectiveFunction,
    VariationalSolver,
)

from usability_oracle.variational.kl_divergence import (
    compute_kl_divergence,
    compute_kl_discrete,
    compute_kl_gaussian,
    compute_mutual_information,
    compute_policy_kl,
    renyi_divergence,
    symmetric_kl,
)

from usability_oracle.variational.free_energy import (
    FreeEnergyComputer,
    compute_free_energy,
    compute_optimal_beta,
    compute_policy_gradient,
    compute_softmax_policy,
    compute_value_iteration,
)

from usability_oracle.variational.capacity import (
    CapacityEstimatorImpl,
    blahut_arimoto,
    compose_capacities,
    estimate_fitts_capacity,
    estimate_hick_capacity,
    estimate_memory_capacity,
    estimate_visual_capacity,
)

from usability_oracle.variational.convergence import (
    ConvergenceMonitor,
    ConvergenceRateType,
    check_convergence,
    compute_convergence_rate,
    detect_oscillation,
    extrapolate_convergence,
    lyapunov_stability,
)

from usability_oracle.variational.optimizer import (
    VariationalOptimizer,
    alternating_minimization,
    line_search,
    mirror_descent,
    natural_gradient_descent,
    proximal_point,
)

__all__ = [
    # types
    "CapacityProfile",
    "ConvergenceInfo",
    "ConvergenceStatus",
    "FreeEnergyResult",
    "KLDivergenceResult",
    "ObjectiveType",
    "VariationalConfig",
    # protocols
    "CapacityEstimator",
    "ObjectiveFunction",
    "VariationalSolver",
    # kl_divergence
    "compute_kl_divergence",
    "compute_kl_discrete",
    "compute_kl_gaussian",
    "compute_mutual_information",
    "compute_policy_kl",
    "renyi_divergence",
    "symmetric_kl",
    # free_energy
    "FreeEnergyComputer",
    "compute_free_energy",
    "compute_optimal_beta",
    "compute_policy_gradient",
    "compute_softmax_policy",
    "compute_value_iteration",
    # capacity
    "CapacityEstimatorImpl",
    "blahut_arimoto",
    "compose_capacities",
    "estimate_fitts_capacity",
    "estimate_hick_capacity",
    "estimate_memory_capacity",
    "estimate_visual_capacity",
    # convergence
    "ConvergenceMonitor",
    "ConvergenceRateType",
    "check_convergence",
    "compute_convergence_rate",
    "detect_oscillation",
    "extrapolate_convergence",
    "lyapunov_stability",
    # optimizer
    "VariationalOptimizer",
    "alternating_minimization",
    "line_search",
    "mirror_descent",
    "natural_gradient_descent",
    "proximal_point",
]
