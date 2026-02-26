"""GP-based surrogate and acquisition functions for adaptive exploration."""

from .surrogate import GPSurrogate, GPPrediction, matern52_kernel
from .acquisition import (
    expected_improvement, upper_confidence_bound,
    boundary_uncertainty, phase_boundary_score,
    AcquisitionOptimizer,
)

__all__ = [
    'GPSurrogate', 'GPPrediction', 'matern52_kernel',
    'expected_improvement', 'upper_confidence_bound',
    'boundary_uncertainty', 'phase_boundary_score',
    'AcquisitionOptimizer',
]
