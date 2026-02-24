"""
Random Matrix Theory module for NTK spectral analysis.

Implements Marchenko-Pastur law, Tracy-Widom edge statistics,
spiked random matrix models, and free probability tools for
analyzing the spectral properties of neural tangent kernels.
"""

from .marchenko_pastur import (
    StieltjesTransform,
    MarchenkoPasturLaw,
    ResolventComputer,
    FreeConvolution,
    BulkEigenvaluePrediction,
)
from .tracy_widom_full import (
    TracyWidomDistribution,
    EdgeFluctuation,
    SpectralEdgeAnalyzer,
    AiryKernelComputer,
)
from .spiked_models import (
    BBPTransition,
    SpikedCovarianceModel,
    SignalDetectionThreshold,
    SpikedNTKModel,
    PlantedFeatureDetector,
)
from .free_probability import (
    RTransform,
    STransform,
    FreeConvolutionEngine,
    MultiplicativeFreeConvolution,
    LayeredNTKSpectrum,
    FreeDeconvolution,
)
from .spectral_analysis import (
    EmpiricalSpectralDistribution,
    WignerSemicircle,
    SpectralRigidity,
    SpectralFlowAnalysis,
    NTKSpectrumAnalyzer,
    MatrixEnsembleGenerator,
    SpectralComparisonTool,
)

__all__ = [
    "StieltjesTransform", "MarchenkoPasturLaw", "ResolventComputer",
    "FreeConvolution", "BulkEigenvaluePrediction",
    "TracyWidomDistribution", "EdgeFluctuation", "SpectralEdgeAnalyzer",
    "AiryKernelComputer",
    "BBPTransition", "SpikedCovarianceModel", "SignalDetectionThreshold",
    "SpikedNTKModel", "PlantedFeatureDetector",
    "RTransform", "STransform", "FreeConvolutionEngine",
    "MultiplicativeFreeConvolution", "LayeredNTKSpectrum", "FreeDeconvolution",
    "EmpiricalSpectralDistribution", "WignerSemicircle", "SpectralRigidity",
    "SpectralFlowAnalysis", "NTKSpectrumAnalyzer", "MatrixEnsembleGenerator",
    "SpectralComparisonTool",
]
