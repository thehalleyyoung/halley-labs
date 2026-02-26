"""
Bisimulation computation module for CoaCert-TLA.

Provides partition refinement algorithms for computing bisimulation
equivalences on labeled transition systems and F-coalgebras:

  - :class:`BisimulationRelation`: union-find equivalence relation
  - :class:`PartitionRefinement`: Paige–Tarjan O(m log n) refinement
  - :class:`StutteringBisimulation`: Groote–Vaandrager stuttering bisimulation
  - :class:`QuotientBuilder`: quotient system construction
  - :class:`FairnessEquivalence`: fairness-respecting equivalence
  - :class:`RefinementEngine`: full pipeline orchestration
"""

from .relation import (
    BisimulationRelation,
    EquivClassInfo,
    RelationStats,
)
from .partition_refinement import (
    Block,
    PartitionRefinement,
    RefinementResult,
    RefinementStep,
)
from .stuttering import (
    StutteringBisimulation,
    StutteringResult,
    StutterCounterexample,
)
from .quotient import (
    QuotientBuilder,
    QuotientStats,
    QuotientVerificationResult,
)
from .fairness_equiv import (
    FairnessEquivalence,
    FairCycleInfo,
    FairEquivalenceResult,
    FairnessVerificationResult,
)
from .refinement_iteration import (
    ComparisonResult,
    EngineResult,
    PhaseRecord,
    RefinementEngine,
    RefinementStrategy,
    RoundRecord,
)

__all__ = [
    # relation
    "BisimulationRelation",
    "EquivClassInfo",
    "RelationStats",
    # partition refinement
    "Block",
    "PartitionRefinement",
    "RefinementResult",
    "RefinementStep",
    # stuttering
    "StutteringBisimulation",
    "StutteringResult",
    "StutterCounterexample",
    # quotient
    "QuotientBuilder",
    "QuotientStats",
    "QuotientVerificationResult",
    # fairness
    "FairnessEquivalence",
    "FairCycleInfo",
    "FairEquivalenceResult",
    "FairnessVerificationResult",
    # engine
    "ComparisonResult",
    "EngineResult",
    "PhaseRecord",
    "RefinementEngine",
    "RefinementStrategy",
    "RoundRecord",
]
