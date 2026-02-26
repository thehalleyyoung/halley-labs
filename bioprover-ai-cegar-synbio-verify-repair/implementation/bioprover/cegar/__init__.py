"""
CEGAR (Counterexample-Guided Abstraction Refinement) loop for BioProver.

Implements biology-aware CEGAR with structural, monotonicity, and
time-scale refinement strategies for gene regulatory network verification.
"""

from bioprover.cegar.abstraction import (
    AbstractState,
    AbstractionDomain,
    IntervalAbstraction,
    PredicateAbstraction,
    ProductAbstraction,
)
from bioprover.cegar.cegar_engine import (
    CEGARConfig,
    CEGAREngine,
    CEGARStatistics,
    VerificationResult,
    VerificationStatus,
)
from bioprover.cegar.convergence import (
    ConvergenceMetrics,
    ConvergenceMonitor,
    TerminationReason,
)
from bioprover.cegar.counterexample import (
    AbstractCounterexample,
    ConcreteCounterexample,
    SpuriousnessChecker,
    SpuriousnessResult,
)
from bioprover.cegar.predicate_manager import (
    Predicate,
    PredicateSet,
    PredicateTemplate,
)
from bioprover.cegar.refinement import (
    AIGuidedRefinement,
    InterpolationRefinement,
    MonotonicityRefinement,
    RefinementCombinator,
    RefinementStrategy,
    SimulationGuidedRefinement,
    StructuralRefinement,
    TimeScaleRefinement,
)

__all__ = [
    # Abstraction
    "AbstractState",
    "AbstractionDomain",
    "IntervalAbstraction",
    "PredicateAbstraction",
    "ProductAbstraction",
    # Counterexample
    "AbstractCounterexample",
    "ConcreteCounterexample",
    "SpuriousnessChecker",
    "SpuriousnessResult",
    # Refinement
    "AIGuidedRefinement",
    "RefinementStrategy",
    "StructuralRefinement",
    "MonotonicityRefinement",
    "TimeScaleRefinement",
    "InterpolationRefinement",
    "SimulationGuidedRefinement",
    "RefinementCombinator",
    # Convergence
    "ConvergenceMonitor",
    "ConvergenceMetrics",
    "TerminationReason",
    # Engine
    "CEGAREngine",
    "CEGARConfig",
    "CEGARStatistics",
    "VerificationResult",
    "VerificationStatus",
    # Predicates
    "Predicate",
    "PredicateSet",
    "PredicateTemplate",
]
