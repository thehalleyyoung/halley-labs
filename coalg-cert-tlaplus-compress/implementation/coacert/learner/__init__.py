"""
L*-style coalgebraic learning engine for CoaCert-TLA.

Implements Angluin-style active learning adapted for F-coalgebras where
F(X) = P(AP) × P(X)^Act × Fair(X).  The learner iteratively builds an
observation table, poses membership and equivalence queries, and converges
to a minimal hypothesis coalgebra that is behaviourally equivalent to the
concrete system.

Modules
-------
observation_table
    Observation table data structure indexed by access sequences and
    distinguishing suffixes.
membership_oracle
    Membership query engine that executes action sequences on a concrete
    transition system.
equivalence_oracle
    Bounded conformance testing for hypothesis equivalence checking.
learner
    Main L*-style learning loop with closedness/consistency repair.
hypothesis
    Hypothesis coalgebra construction from a closed, consistent table.
counterexample
    Counterexample processing with Rivest-Schapire optimisation.
convergence
    Convergence analysis and complexity bound computation.
"""

from .observation_table import ObservationTable
from .membership_oracle import MembershipOracle
from .equivalence_oracle import EquivalenceOracle
from .learner import CoalgebraicLearner
from .hypothesis import HypothesisBuilder
from .counterexample import CounterexampleProcessor
from .convergence import ConvergenceAnalyzer
from .diameter_computation import (
    ExactDiameterComputer,
    IncrementalDiameter,
    DiameterCertificate,
)
from .incremental_deepening import (
    IncrementalDeepeningOracle,
    ConvergenceCertificate,
    ConvergenceHistory,
    DeepeningRound,
)
from .w_method import WMethodTester, WMethodResult

from .conformance_gap import (
    ConformanceGapAnalyzer,
    ConformanceCompleteCertificate,
    compute_sufficient_depth,
)

__all__ = [
    "ObservationTable",
    "MembershipOracle",
    "EquivalenceOracle",
    "CoalgebraicLearner",
    "HypothesisBuilder",
    "CounterexampleProcessor",
    "ConvergenceAnalyzer",
    "ExactDiameterComputer",
    "IncrementalDiameter",
    "DiameterCertificate",
    "IncrementalDeepeningOracle",
    "ConvergenceCertificate",
    "ConvergenceHistory",
    "DeepeningRound",
    "WMethodTester",
    "WMethodResult",
    "ConformanceGapAnalyzer",
    "ConformanceCompleteCertificate",
    "compute_sufficient_depth",
]
