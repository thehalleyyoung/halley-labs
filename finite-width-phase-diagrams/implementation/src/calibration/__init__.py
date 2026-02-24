"""Multi-width calibration for finite-width NTK corrections.

Provides regression, bootstrap confidence intervals, and a full
calibration pipeline for extracting 1/N correction coefficients.
"""

from .regression import (
    CalibrationRegression,
    RegressionResult,
    DesignMatrixBuilder,
    ConstrainedRegression,
)
from .bootstrap import (
    BootstrapCI,
    BootstrapResult,
    BoundaryUncertainty,
)
from .pipeline import (
    CalibrationPipeline,
    CalibrationResult,
    CalibrationConfig,
)
