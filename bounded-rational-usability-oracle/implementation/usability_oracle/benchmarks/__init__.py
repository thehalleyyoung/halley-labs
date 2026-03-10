"""
usability_oracle.benchmarks — Benchmark suite for usability oracle evaluation.

Provides synthetic UI generation, mutation-based bottleneck injection,
dataset management, and metrics computation for evaluating the oracle
against ground truth.
"""

from __future__ import annotations

from usability_oracle.benchmarks.suite import (
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkSuite,
)
from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationGenerator
from usability_oracle.benchmarks.datasets import DatasetManager
from usability_oracle.benchmarks.metrics import BenchmarkMetrics

__all__ = [
    "BenchmarkCase",
    "BenchmarkReport",
    "BenchmarkResult",
    "BenchmarkSuite",
    "SyntheticUIGenerator",
    "MutationGenerator",
    "DatasetManager",
    "BenchmarkMetrics",
]
