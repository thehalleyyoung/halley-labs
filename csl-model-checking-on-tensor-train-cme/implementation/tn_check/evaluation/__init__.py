"""
Evaluation and benchmarking framework.

Provides ground-truth comparison for small systems and
TT-compressibility survey utilities.
"""

from tn_check.evaluation.benchmark import (
    ground_truth_check,
    compressibility_survey,
    BenchmarkResult,
)

__all__ = [
    "ground_truth_check",
    "compressibility_survey",
    "BenchmarkResult",
]
