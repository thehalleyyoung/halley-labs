"""
Mean Field Theory module for finite-width neural network phase diagrams.

Implements self-consistent mean-field equations, free energy landscapes,
susceptibility computations, and replica methods for analyzing neural
network phase transitions in the infinite-width limit and finite-width
corrections thereof.
"""

from .order_parameters import (
    OrderParameterSolver,
    OverlapParameter,
    CorrelationFunction,
    ResponseFunction,
    FixedPointIterator,
    MultiFixedPointDetector,
    FixedPointStabilityAnalyzer,
)
from .free_energy import (
    FreeEnergyLandscape,
    SaddlePointSolver,
    FreeEnergyBarrier,
    PhaseTransitionDetector,
    TransitionClassifier,
)
from .susceptibility import (
    LinearResponseComputer,
    FluctuationDissipation,
    DynamicSusceptibility,
    CriticalExponentExtractor,
    FiniteSizeScaling,
)
from .replica import (
    ReplicaSymmetricSolver,
    OneStepRSBSolver,
    DeAlmeidaThoulessChecker,
    OverlapDistribution,
    ParisiFunctional,
)
from .signal_propagation import (
    PropagationConfig,
    ActivationKernels,
    ForwardPropagation,
    BackwardPropagation,
    FixedPointAnalyzer,
    CriticalInitialization,
    DepthPhaseAnalyzer,
)
from .phase_transitions import (
    PhaseTransitionConfig,
    OrderDisorderTransition,
    ChaosTransition,
    InformationPropagation,
    PhaseTransitionAnalyzer,
    InitializationPhaseDiagram,
)
from .statistical_mechanics import (
    ThermodynamicConfig,
    PartitionFunction,
    Entropy,
    MeanFieldThermodynamics,
    NeuralNetworkThermodynamics,
    SpinGlassAnalogy,
)

__all__ = [
    "OrderParameterSolver",
    "OverlapParameter",
    "CorrelationFunction",
    "ResponseFunction",
    "FixedPointIterator",
    "MultiFixedPointDetector",
    "FixedPointStabilityAnalyzer",
    "FreeEnergyLandscape",
    "SaddlePointSolver",
    "FreeEnergyBarrier",
    "PhaseTransitionDetector",
    "TransitionClassifier",
    "LinearResponseComputer",
    "FluctuationDissipation",
    "DynamicSusceptibility",
    "CriticalExponentExtractor",
    "FiniteSizeScaling",
    "ReplicaSymmetricSolver",
    "OneStepRSBSolver",
    "DeAlmeidaThoulessChecker",
    "OverlapDistribution",
    "ParisiFunctional",
    "PropagationConfig",
    "ActivationKernels",
    "ForwardPropagation",
    "BackwardPropagation",
    "FixedPointAnalyzer",
    "CriticalInitialization",
    "DepthPhaseAnalyzer",
    "PhaseTransitionConfig",
    "OrderDisorderTransition",
    "ChaosTransition",
    "InformationPropagation",
    "PhaseTransitionAnalyzer",
    "InitializationPhaseDiagram",
    "ThermodynamicConfig",
    "PartitionFunction",
    "Entropy",
    "MeanFieldThermodynamics",
    "NeuralNetworkThermodynamics",
    "SpinGlassAnalogy",
]
