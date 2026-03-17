"""Benchmarks module: synthetic benchmark generation, running, and evaluation.

Supports planted-solution methodology for rigorous solver comparison.
"""

from regsynth_py.benchmarks.generator import BenchmarkGenerator
from regsynth_py.benchmarks.runner import BenchmarkRunner
from regsynth_py.benchmarks.evaluator import BenchmarkEvaluator
from regsynth_py.benchmarks.scalability import ScalabilityAnalyzer

__all__ = [
    "BenchmarkGenerator", "BenchmarkRunner",
    "BenchmarkEvaluator", "ScalabilityAnalyzer",
]
