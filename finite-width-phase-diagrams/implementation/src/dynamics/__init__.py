"""
Training Dynamics Simulator for neural network phase diagram analysis.

Implements gradient flow ODEs, lazy/kernel regime analysis, rich/feature-learning
regime tracking, SGD dynamics simulation, and loss landscape analysis for
understanding training-time phase transitions.
"""

from .gradient_flow import (
    GradientFlowSolver,
    NTKDynamics,
    FeatureLearningDynamics,
    LearningRateScheduler,
    GradientNoiseSDE,
)
from .lazy_regime import (
    LazyRegimeAnalyzer,
    NTKStabilityChecker,
    LinearizedDynamicsSolver,
    KernelRegressionPredictor,
    LazyToRichTransitionDetector,
)
from .rich_regime import (
    RichRegimeAnalyzer,
    FeatureEvolutionTracker,
    RepresentationChangeMetric,
    FeatureAlignmentAnalyzer,
    NeuralCollapseDetector,
)
from .sgd_dynamics import (
    SGDSimulator,
    LearningRatePhaseAnalyzer,
    BatchSizeEffectAnalyzer,
    MomentumDynamics,
    SGDNoiseCovarianceEstimator,
    SGDtoSDEConverter,
)
from .loss_landscape import (
    HessianAnalyzer,
    LossSurfaceVisualizer,
    SaddlePointDetector,
    LossBarrierEstimator,
    TrajectoryAnalyzer,
)
from .dynamical_critical import (
    DynamicalCriticalConfig,
    CriticalSlowingDown,
    DynamicExponents,
    KibbleZurekMechanism,
)
from .bifurcation import (
    BifurcationConfig,
    SaddleNodeBifurcation,
    HopfBifurcation,
    PitchforkBifurcation,
    CenterManifoldReduction,
    TrainingBifurcationAnalyzer,
)
from .aging import (
    AgingConfig,
    TwoTimeCorrelation,
    FluctuationDissipationViolation,
    GrokingConnection,
)
from .training_phases import (
    TrainingPhaseConfig,
    PhaseIdentifier,
    CatapultPhaseAnalysis,
    EdgeOfStabilityAnalysis,
    CondensationAnalysis,
    DoubleDescentAnalysis,
)

__all__ = [
    "GradientFlowSolver",
    "NTKDynamics",
    "FeatureLearningDynamics",
    "LearningRateScheduler",
    "GradientNoiseSDE",
    "LazyRegimeAnalyzer",
    "NTKStabilityChecker",
    "LinearizedDynamicsSolver",
    "KernelRegressionPredictor",
    "LazyToRichTransitionDetector",
    "RichRegimeAnalyzer",
    "FeatureEvolutionTracker",
    "RepresentationChangeMetric",
    "FeatureAlignmentAnalyzer",
    "NeuralCollapseDetector",
    "SGDSimulator",
    "LearningRatePhaseAnalyzer",
    "BatchSizeEffectAnalyzer",
    "MomentumDynamics",
    "SGDNoiseCovarianceEstimator",
    "SGDtoSDEConverter",
    "HessianAnalyzer",
    "LossSurfaceVisualizer",
    "SaddlePointDetector",
    "LossBarrierEstimator",
    "TrajectoryAnalyzer",
    "DynamicalCriticalConfig",
    "CriticalSlowingDown",
    "DynamicExponents",
    "KibbleZurekMechanism",
    "BifurcationConfig",
    "SaddleNodeBifurcation",
    "HopfBifurcation",
    "PitchforkBifurcation",
    "CenterManifoldReduction",
    "TrainingBifurcationAnalyzer",
    "AgingConfig",
    "TwoTimeCorrelation",
    "FluctuationDissipationViolation",
    "GrokingConnection",
    "TrainingPhaseConfig",
    "PhaseIdentifier",
    "CatapultPhaseAnalysis",
    "EdgeOfStabilityAnalysis",
    "CondensationAnalysis",
    "DoubleDescentAnalysis",
]
