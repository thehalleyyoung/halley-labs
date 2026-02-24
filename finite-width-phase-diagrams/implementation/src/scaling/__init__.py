"""
Scaling Laws module for neural network phase diagram analysis.

Implements µP (maximal update parameterization), width and depth scaling,
and neural scaling law fitting/prediction for understanding how phase
boundaries and critical behavior scale with network dimensions.
"""

from .mu_p import (
    MuPScalingComputer,
    MuPLearningRateTransfer,
    MuPInitialization,
    WidthIndependenceVerifier,
    MuPViolationDetector,
    CoordinateCheck,
)
from .width_scaling import (
    NTKWidthScaling,
    FiniteWidthCorrectionScaling,
    CriticalExponentExtractor,
    ScalingCollapseAnalyzer,
    PowerLawFitter,
)
from .depth_scaling import (
    KernelDepthPropagation,
    SignalPropagationAnalyzer,
    DepthPhaseBoundary,
    OptimalDepthPredictor,
    DepthWidthInteraction,
)
from .scaling_laws import (
    ScalingExponentComputer,
    ScalingLawFitter,
    ScalingLawPredictor,
    ChinchillaAllocator,
    ArchitectureScalingComparator,
)
from .universality import (
    UniversalityAnalyzer,
    CriticalExponents,
    UniversalityResult,
)
from .rg_flow import (
    RGConfig,
    BlockSpinTransformation,
    BetaFunction,
    RGFixedPoint,
    RGFlow,
    UniversalityFromRG,
)
from .finite_size_scaling import (
    FiniteSizeScalingConfig,
    ScalingCollapseEngine,
    CriticalPointExtractor,
    CriticalExponentMeasurer,
    NeuralNetworkFSS,
)

__all__ = [
    "MuPScalingComputer",
    "MuPLearningRateTransfer",
    "MuPInitialization",
    "WidthIndependenceVerifier",
    "MuPViolationDetector",
    "CoordinateCheck",
    "NTKWidthScaling",
    "FiniteWidthCorrectionScaling",
    "CriticalExponentExtractor",
    "ScalingCollapseAnalyzer",
    "PowerLawFitter",
    "KernelDepthPropagation",
    "SignalPropagationAnalyzer",
    "DepthPhaseBoundary",
    "OptimalDepthPredictor",
    "DepthWidthInteraction",
    "ScalingExponentComputer",
    "ScalingLawFitter",
    "ScalingLawPredictor",
    "ChinchillaAllocator",
    "ArchitectureScalingComparator",
    "UniversalityAnalyzer",
    "CriticalExponents",
    "UniversalityResult",
    "RGConfig",
    "BlockSpinTransformation",
    "BetaFunction",
    "RGFixedPoint",
    "RGFlow",
    "UniversalityFromRG",
    "FiniteSizeScalingConfig",
    "ScalingCollapseEngine",
    "CriticalPointExtractor",
    "CriticalExponentMeasurer",
    "NeuralNetworkFSS",
]
