"""Phase mapper for constructing finite-width neural network phase diagrams.

Provides grid sweeps over hyperparameter space, pseudo-arclength continuation
for tracking phase boundaries, boundary extraction, order parameter computation,
and phase diagram data structures.
"""

from .grid_sweep import (
    GridConfig,
    GridPoint,
    GridSweeper,
    ParameterRange,
    SweepResult,
)
from .continuation import (
    BranchInfo,
    ContinuationConfig,
    ContinuationPoint,
    ContinuationResult,
    PseudoArclengthContinuation,
)
from .boundary import (
    BoundaryConfig,
    BoundaryCurve,
    BoundaryExtractor,
    BoundaryPoint,
)
from .order_parameter import (
    OrderParameterComputer,
    OrderParameterResult,
    OrderParameterType,
    TrainingTrajectory,
)
from .phase_diagram import (
    PhaseDiagram,
    RegimeRegion,
    RegimeType,
)
from .gamma_star import (
    GammaStarResult,
    PhaseBoundaryPredictor,
)
from .multi_task import (
    MultiTaskConfig,
    TaskInterferenceAnalyzer,
    MultiTaskPhaseBoundary,
    TransferLearningPhases,
    MultiTaskSpectralAnalysis,
    MultiTaskExperiment,
)
