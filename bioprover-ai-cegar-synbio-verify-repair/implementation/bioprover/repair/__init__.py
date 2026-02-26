"""Parameter repair and synthesis module for BioProver.

Provides CEGIS inner loop, STL robustness maximization, design space
exploration with Pareto frontier computation, biological realizability
constraints, and repair report generation.
"""

from bioprover.repair.cegis import (
    CEGISLoop,
    CEGISConfig,
    CEGISResult,
    CEGISStatus,
    CounterexampleSet,
    ProposalStrategy,
)
from bioprover.repair.robustness_optimization import (
    CMAES,
    CMAESConfig,
    RobustnessOptimizer,
    OptimizationResult,
)
from bioprover.repair.design_space import (
    DesignSpace,
    DesignPoint,
    ParetoFrontier,
    SensitivityResult,
)
from bioprover.repair.realizability import (
    RealizabilityChecker,
    RealizabilityReport,
    ConstraintViolation,
    BiologicalConstraint,
)
from bioprover.repair.repair_report import (
    RepairResult,
    RepairReport,
    ReportFormat,
)
from bioprover.repair.parameter_synthesis import (
    ParameterSynthesizer,
    SynthesisConfig,
    SynthesisMode,
    SynthesisResult,
)

__all__ = [
    # cegis
    "CEGISLoop",
    "CEGISConfig",
    "CEGISResult",
    "CEGISStatus",
    "CounterexampleSet",
    "ProposalStrategy",
    # robustness
    "CMAES",
    "CMAESConfig",
    "RobustnessOptimizer",
    "OptimizationResult",
    # design space
    "DesignSpace",
    "DesignPoint",
    "ParetoFrontier",
    "SensitivityResult",
    # realizability
    "RealizabilityChecker",
    "RealizabilityReport",
    "ConstraintViolation",
    "BiologicalConstraint",
    # report
    "RepairResult",
    "RepairReport",
    "ReportFormat",
    # synthesis
    "ParameterSynthesizer",
    "SynthesisConfig",
    "SynthesisMode",
    "SynthesisResult",
]
