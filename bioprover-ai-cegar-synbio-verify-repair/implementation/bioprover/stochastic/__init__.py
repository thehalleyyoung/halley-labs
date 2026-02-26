"""
Stochastic simulation engine for BioProver.

Provides SSA/Gillespie variants, tau-leaping, finite state projection,
moment closure approximations, hybrid SSA/ODE, and parallel ensemble management.
"""

from .ssa import (
    Reaction,
    StochasticState,
    DirectMethod,
    NextReactionMethod,
    TrajectoryRecorder,
    run_ensemble_ssa,
)
from .tau_leaping import (
    ExplicitTauLeaping,
    ImplicitTauLeaping,
    MidpointTauLeaping,
    AdaptiveTauSelector,
    SSATauLeapingSwitch,
    StepSizeController,
)
from .fsp import (
    FSPSolver,
    StateSpace,
    SparseTransitionMatrix,
    MarginalDistribution,
)
from .moment_closure import (
    MomentEquations,
    NormalClosure,
    LogNormalClosure,
    ZeroCumulantClosure,
    DerivativeMatchingClosure,
    LinearNoiseApproximation,
    ClosureComparison,
)
from .hybrid import (
    SpeciesPartition,
    HaseltineRawlingsHybrid,
    DynamicRepartitioner,
    HybridTrajectory,
)
from .ensemble import (
    EnsembleSimulator,
    EnsembleStatistics,
    ConvergenceDetector,
    TrajectoryStore,
)

__all__ = [
    "Reaction",
    "StochasticState",
    "DirectMethod",
    "NextReactionMethod",
    "TrajectoryRecorder",
    "run_ensemble_ssa",
    "ExplicitTauLeaping",
    "ImplicitTauLeaping",
    "MidpointTauLeaping",
    "AdaptiveTauSelector",
    "SSATauLeapingSwitch",
    "StepSizeController",
    "FSPSolver",
    "StateSpace",
    "SparseTransitionMatrix",
    "MarginalDistribution",
    "MomentEquations",
    "NormalClosure",
    "LogNormalClosure",
    "ZeroCumulantClosure",
    "DerivativeMatchingClosure",
    "LinearNoiseApproximation",
    "ClosureComparison",
    "SpeciesPartition",
    "HaseltineRawlingsHybrid",
    "DynamicRepartitioner",
    "HybridTrajectory",
    "EnsembleSimulator",
    "EnsembleStatistics",
    "ConvergenceDetector",
    "TrajectoryStore",
]
