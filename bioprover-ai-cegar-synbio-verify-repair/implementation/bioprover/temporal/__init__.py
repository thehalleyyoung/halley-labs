"""BioProver Temporal Logic Engine.

Provides Bio-STL parsing, quantitative robustness semantics,
interval model checking, statistical model checking, and
bounded model checking with k-induction.
"""

from bioprover.temporal.stl_ast import (
    STLFormula,
    Predicate,
    STLAnd,
    STLOr,
    STLNot,
    STLImplies,
    Always,
    Eventually,
    Until,
    Interval,
)

from bioprover.temporal.bio_stl_parser import BioSTLParser, ParseError

from bioprover.temporal.robustness import (
    Signal,
    RobustnessComputer,
    compute_robustness,
    sensitivity_analysis,
)

from bioprover.temporal.interval_model_checking import (
    ThreeValued,
    ThreeValuedResult,
    IntervalSignal,
    IntervalModelChecker,
)

from bioprover.temporal.statistical_model_checking import (
    SPRTChecker,
    BayesianEstimator,
    StatisticalModelChecker,
)

from bioprover.temporal.bmc import (
    BMCEncoder,
    KInduction,
    InvariantTemplate,
    BMCResult,
)

__all__ = [
    # AST
    "STLFormula",
    "Predicate",
    "STLAnd",
    "STLOr",
    "STLNot",
    "STLImplies",
    "Always",
    "Eventually",
    "Until",
    "Interval",
    # Parser
    "BioSTLParser",
    "ParseError",
    # Robustness
    "Signal",
    "RobustnessComputer",
    "compute_robustness",
    "sensitivity_analysis",
    # Interval model checking
    "ThreeValued",
    "ThreeValuedResult",
    "IntervalSignal",
    "IntervalModelChecker",
    # Statistical model checking
    "SPRTChecker",
    "BayesianEstimator",
    "StatisticalModelChecker",
    # BMC
    "BMCEncoder",
    "KInduction",
    "InvariantTemplate",
    "BMCResult",
]
