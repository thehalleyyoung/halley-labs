"""Finite-width corrections for NTK phase diagram computation.

Provides 1/N expansion corrections, H-tensor computation,
and perturbative validity analysis for finite-width neural networks.
"""

from .finite_width import FiniteWidthCorrector, CorrectionResult, ConvergenceInfo
from .h_tensor import HTensor, HTensorComputer, FactorizationValidator
from .perturbation import (
    PerturbativeValidator,
    ValidityResult,
    ConfidenceLevel,
    ConvergenceRadius,
)
from .trace_normalized import (
    TraceNormalizedCorrector,
    NormalizedCorrectionResult,
    PadeResummer,
)
from .nonperturbative import (
    SaddlePointApproximation,
    InstantonCalculator,
    BorelResummation,
    PathIntegralNN,
    NumericalBootstrap,
)
from .resurgence import (
    TransSeries,
    LargeOrderAnalysis,
    PadeBorelAnalysis,
    ResurgentNTKCorrections,
)
