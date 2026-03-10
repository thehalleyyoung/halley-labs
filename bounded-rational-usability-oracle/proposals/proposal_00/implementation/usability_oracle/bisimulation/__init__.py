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

__all__ = [
    "Partition",
    "BisimulationResult",
    "CognitiveDistanceMatrix",
    "CognitiveDistanceComputer",
    "PartitionRefinement",
    "QuotientMDPBuilder",
    "FeatureBasedClustering",
    "BisimulationValidator",
]
