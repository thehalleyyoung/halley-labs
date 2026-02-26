"""
CoaCert-TLA: Coalgebraic Certified Compression for TLA+ Specifications.

Implements F-coalgebra-based bisimulation quotient compression with
Merkle-hashed witnesses for verifiable state-space reduction.
"""

__version__ = "0.1.0"
__author__ = "CoaCert-TLA Contributors"

# Parser: TLA-lite lexing, parsing, type-checking
from coacert.parser import parse, Parser, TypeChecker, PrettyPrinter, Module

# Semantics: expression evaluation, state dynamics
from coacert.semantics import (
    evaluate,
    TLAState,
    ActionEvaluator,
    Environment,
    compute_successors,
    compute_initial_states,
)

# Explorer: explicit-state model exploration
from coacert.explorer import (
    ExplicitStateExplorer,
    TransitionGraph,
    ExplorationStats,
    SymmetryDetector,
)

# Functor: polynomial functor F(X) = P(AP) × P(X)^Act × Fair(X)
from coacert.functor import (
    FCoalgebra,
    PowersetFunctor,
    StutterMonad,
    TFairCoherenceChecker,
    PartitionRefinement as FunctorPartitionRefinement,
)

# Learner: L*-style active automata learning for coalgebras
from coacert.learner import (
    CoalgebraicLearner,
    ObservationTable,
    MembershipOracle,
    EquivalenceOracle,
)

# Bisimulation: partition refinement, quotient construction
from coacert.bisimulation import (
    RefinementEngine,
    PartitionRefinement,
    BisimulationRelation,
    QuotientBuilder,
)

# Witness: Merkle-hashed bisimulation certificates
from coacert.witness import MerkleTree, TransitionWitness, CompactWitness

# Verifier: standalone witness verification
from coacert.verifier import verify_witness, VerificationReport

# Properties: CTL*/LTL model checking, differential testing
from coacert.properties import (
    CTLStarChecker,
    SafetyChecker,
    DifferentialTester,
    parse_formula,
)

# Specs: built-in benchmark specifications
from coacert.specs import SpecRegistry, TwoPhaseCommitSpec, PaxosSpec

# Evaluation: benchmarking and metrics
from coacert.evaluation import BenchmarkRunner, MetricsCollector

__all__ = [
    # Core pipeline
    "parse", "Parser", "TypeChecker", "PrettyPrinter", "Module",
    "evaluate", "TLAState", "ActionEvaluator", "Environment",
    "compute_successors", "compute_initial_states",
    "ExplicitStateExplorer", "TransitionGraph", "ExplorationStats",
    "FCoalgebra", "PowersetFunctor", "StutterMonad",
    "CoalgebraicLearner", "ObservationTable",
    "MembershipOracle", "EquivalenceOracle",
    "RefinementEngine", "PartitionRefinement", "BisimulationRelation",
    "QuotientBuilder",
    "MerkleTree", "TransitionWitness", "CompactWitness",
    "verify_witness", "VerificationReport",
    "CTLStarChecker", "SafetyChecker", "DifferentialTester", "parse_formula",
    "SpecRegistry", "TwoPhaseCommitSpec", "PaxosSpec",
    "BenchmarkRunner", "MetricsCollector",
]