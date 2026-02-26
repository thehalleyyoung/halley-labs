"""
SMT certificate generation and δ-bound computation.

Provides:
- δ-bound computation for soundness of δ-complete SMT encoding
- SMT-LIB2 formula generation for regime claims
- Interface to dReal and Z3 solvers
"""

from .delta_bound import (
    DeltaBound,
    DeltaBoundResult,
    compute_required_delta,
    compute_eigenvalue_gap,
    soundness_margin,
)

__all__ = [
    "DeltaBound",
    "DeltaBoundResult",
    "compute_required_delta",
    "compute_eigenvalue_gap",
    "soundness_margin",
]
