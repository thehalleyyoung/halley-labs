"""
Bounded-Rational Usability Oracle
==================================

An information-theoretic bounded-rational usability regression testing system.

Uses free-energy formulation F(π) = E_π[R] - (1/β) D_KL(π || p₀) to unify
cognitive laws (Fitts', Hick-Hyman, visual search, working memory) into a single
variational objective. The system parses UI accessibility trees into MDPs,
computes cognitive costs, and detects usability regressions.

Core modules
------------
- **core**: Fundamental types, enumerations, protocols, errors, config, constants
- **accessibility**: Accessibility tree parsing and representation
- **alignment**: Tree alignment / edit-distance computation
- **cognitive**: Cognitive cost models (Fitts, Hick-Hyman, visual search, WM)
- **mdp**: MDP construction from accessibility trees
- **policy**: Bounded-rational policy computation via free-energy minimisation
- **bisimulation**: State-space reduction via bisimulation quotients
- **comparison**: Statistical comparison of usability metrics
- **bottleneck**: Cognitive bottleneck classification
- **repair**: Automated repair synthesis for usability regressions
- **pipeline**: End-to-end orchestration of pipeline stages
- **output**: Formatting (JSON, SARIF, HTML, console)
- **cli**: Command-line interface
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Bounded-Rational Usability Oracle Contributors"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Lazy top-level imports — keep startup fast, resolve on first access
# ---------------------------------------------------------------------------

from usability_oracle.core.types import (
    Point2D,
    BoundingBox,
    Interval,
    CostTuple,
    TrajectoryStep,
    Trajectory,
    PolicyDistribution,
)
from usability_oracle.core.enums import (
    AccessibilityRole,
    BottleneckType,
    CognitiveLaw,
    EditOperationType,
    RegressionVerdict,
    PipelineStage,
    ComparisonMode,
    OutputFormat,
    AlignmentPass,
    Severity,
    MotorChannel,
    PerceptualChannel,
)
from usability_oracle.core.protocols import (
    Parser,
    Aligner,
    CostModel,
    PolicyComputer,
    BottleneckClassifier,
    RepairSynthesizer,
    OutputFormatter,
    PipelineStageExecutor,
    Validator,
    Serializable,
    CacheProvider,
)
from usability_oracle.core.errors import (
    UsabilityOracleError,
    ParseError,
    AlignmentError,
    CostModelError,
    MDPError,
    PolicyError,
    BisimulationError,
    BottleneckError,
    ComparisonError,
    RepairError,
    ConfigError,
    PipelineError,
)
from usability_oracle.core.config import OracleConfig

__all__ = [
    # version
    "__version__",
    # types
    "Point2D",
    "BoundingBox",
    "Interval",
    "CostTuple",
    "TrajectoryStep",
    "Trajectory",
    "PolicyDistribution",
    # enums
    "AccessibilityRole",
    "BottleneckType",
    "CognitiveLaw",
    "EditOperationType",
    "RegressionVerdict",
    "PipelineStage",
    "ComparisonMode",
    "OutputFormat",
    "AlignmentPass",
    "Severity",
    "MotorChannel",
    "PerceptualChannel",
    # protocols
    "Parser",
    "Aligner",
    "CostModel",
    "PolicyComputer",
    "BottleneckClassifier",
    "RepairSynthesizer",
    "OutputFormatter",
    "PipelineStageExecutor",
    "Validator",
    "Serializable",
    "CacheProvider",
    # errors
    "UsabilityOracleError",
    "ParseError",
    "AlignmentError",
    "CostModelError",
    "MDPError",
    "PolicyError",
    "BisimulationError",
    "BottleneckError",
    "ComparisonError",
    "RepairError",
    "ConfigError",
    "PipelineError",
    # config
    "OracleConfig",
]
