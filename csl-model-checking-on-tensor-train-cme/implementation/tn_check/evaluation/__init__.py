"""
Evaluation and benchmarking framework.

Provides ground-truth comparison for small systems,
TT-compressibility survey utilities, scaling benchmarks,
accuracy benchmarks, and PRISM-style comparison utilities.
"""

from tn_check.evaluation.benchmark import (
    ground_truth_check,
    compressibility_survey,
    BenchmarkResult,
    run_scaling_benchmark,
    run_accuracy_benchmark,
    run_all_benchmarks,
)

__all__ = [
    "ground_truth_check",
    "compressibility_survey",
    "BenchmarkResult",
    "run_scaling_benchmark",
    "run_accuracy_benchmark",
    "run_all_benchmarks",
]
