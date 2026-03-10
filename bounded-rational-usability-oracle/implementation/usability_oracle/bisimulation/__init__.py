"""
usability_oracle.bisimulation — Bounded-rational bisimulation quotients.

Implements state-space reduction calibrated to cognitive capacity β.  Two UI
states are *behaviourally equivalent* under bounded rationality when their
bounded-rational policy distributions are indistinguishable:

    d_cog(s₁, s₂) = sup_{β' ≤ β}  d_TV(π_{β'}(·|s₁), π_{β'}(·|s₂))

The quotient MDP collapses equivalent states, reducing the effective state
space while preserving the usability analysis.

Re-exports
----------
>>> from usability_oracle.bisimulation import (
...     Partition, BisimulationResult, CognitiveDistanceMatrix,
...     CognitiveDistanceComputer, PartitionRefinement,
...     QuotientMDPBuilder, FeatureBasedClustering, BisimulationValidator,
...     LarsenSkouBisimulation, ProbabilisticBisimulationMetric,
...     ApproximateProbabilisticBisimulation, CapacityWeightedMetric,
...     kantorovich_distance, kantorovich_coupling,
...     GraphLaplacian, SpectralBisimulation, LowRankTransitionApproximation,
...     fiedler_partition,
...     SimulationRelation, ReadySimulation, FailureSimulation,
...     ProbabilisticSimulation, SimulationDistance, SimulationGame,
...     BisimulationUpTo, CongruenceChecker, ParallelComposition,
...     ModularReduction,
...     EpsilonBisimulation, ApproximatePartitionRefinement,
...     ErrorPropagation, AdaptiveRefinement, AbstractionTradeoff,
...     PolicySensitivity, FreeEnergyDistance, CognitiveKernel,
...     CognitiveAggregation,
... )
"""

from __future__ import annotations

from usability_oracle.bisimulation.models import (
    Partition,
    BisimulationResult,
    CognitiveDistanceMatrix,
)
from usability_oracle.bisimulation.cognitive_distance import CognitiveDistanceComputer
from usability_oracle.bisimulation.partition import PartitionRefinement
from usability_oracle.bisimulation.quotient import QuotientMDPBuilder
from usability_oracle.bisimulation.clustering import FeatureBasedClustering
from usability_oracle.bisimulation.validators import BisimulationValidator

# Probabilistic bisimulation
from usability_oracle.bisimulation.probabilistic import (
    LarsenSkouBisimulation,
    ProbabilisticBisimulationMetric,
    ApproximateProbabilisticBisimulation,
    CapacityWeightedMetric,
    kantorovich_distance,
    kantorovich_coupling,
)

# Spectral bisimulation
from usability_oracle.bisimulation.spectral import (
    GraphLaplacian,
    SpectralBisimulation,
    LowRankTransitionApproximation,
    fiedler_partition,
)

# Simulation relations
from usability_oracle.bisimulation.simulation_relation import (
    SimulationRelation,
    ReadySimulation,
    FailureSimulation,
    ProbabilisticSimulation,
    SimulationDistance,
    SimulationGame,
)

# Compositional bisimulation
from usability_oracle.bisimulation.compositional import (
    BisimulationUpTo,
    CongruenceChecker,
    ParallelComposition,
    ModularReduction,
)

# Approximate bisimulation
from usability_oracle.bisimulation.approximate import (
    EpsilonBisimulation,
    ApproximatePartitionRefinement,
    ErrorPropagation,
    AdaptiveRefinement,
    AbstractionTradeoff,
)

# Cognitive bisimulation metric
from usability_oracle.bisimulation.cognitive_metric import (
    PolicySensitivity,
    FreeEnergyDistance,
    CognitiveKernel,
    CognitiveAggregation,
)

__all__ = [
    # Core models
    "Partition",
    "BisimulationResult",
    "CognitiveDistanceMatrix",
    # Existing modules
    "CognitiveDistanceComputer",
    "PartitionRefinement",
    "QuotientMDPBuilder",
    "FeatureBasedClustering",
    "BisimulationValidator",
    # Probabilistic bisimulation
    "LarsenSkouBisimulation",
    "ProbabilisticBisimulationMetric",
    "ApproximateProbabilisticBisimulation",
    "CapacityWeightedMetric",
    "kantorovich_distance",
    "kantorovich_coupling",
    # Spectral bisimulation
    "GraphLaplacian",
    "SpectralBisimulation",
    "LowRankTransitionApproximation",
    "fiedler_partition",
    # Simulation relations
    "SimulationRelation",
    "ReadySimulation",
    "FailureSimulation",
    "ProbabilisticSimulation",
    "SimulationDistance",
    "SimulationGame",
    # Compositional bisimulation
    "BisimulationUpTo",
    "CongruenceChecker",
    "ParallelComposition",
    "ModularReduction",
    # Approximate bisimulation
    "EpsilonBisimulation",
    "ApproximatePartitionRefinement",
    "ErrorPropagation",
    "AdaptiveRefinement",
    "AbstractionTradeoff",
    # Cognitive bisimulation metric
    "PolicySensitivity",
    "FreeEnergyDistance",
    "CognitiveKernel",
    "CognitiveAggregation",
]
