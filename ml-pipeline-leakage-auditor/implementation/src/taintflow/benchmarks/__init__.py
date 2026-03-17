"""
taintflow.benchmarks – Synthetic datasets and benchmark scenarios.

This package provides controlled benchmark environments for evaluating
TaintFlow's leakage-detection accuracy.  Every synthetic dataset ships
with a *ground truth* that records the exact location and magnitude of
injected leakage, enabling automated soundness and tightness checks.

Public API
----------
* :class:`SyntheticDataGenerator` – generate datasets with known leakage.
* :class:`DatasetConfig`          – configuration for synthetic datasets.
* :class:`GroundTruth`            – per-feature, per-stage true leakage.
* :class:`BenchmarkRunner`        – run TaintFlow against benchmarks.
* :class:`BenchmarkResult`        – single-scenario result.
* :class:`BenchmarkSuite`         – collection of scenarios.
* :class:`BenchmarkReport`        – summary + per-scenario breakdown.
* :class:`BenchmarkScenario`      – a named, self-contained scenario.
* :class:`StandardBenchmarks`     – library of predefined scenarios.
* :class:`PipelineStep`           – one operation in a simulated pipeline.
* :class:`SimulatedPipeline`      – execute pipeline steps symbolically.
* :class:`ScenarioValidator`      – consistency checks for scenarios.
"""

from __future__ import annotations

from taintflow.benchmarks.synthetic import (
    DatasetConfig,
    GroundTruth,
    SyntheticDataGenerator,
)
from taintflow.benchmarks.runner import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    PerformanceProfiler,
    RegressionDetector,
)
from taintflow.benchmarks.scenarios import (
    BenchmarkScenario,
    PipelineStep,
    ScenarioValidator,
    SimulatedPipeline,
    StandardBenchmarks,
)

__all__: list[str] = [
    "DatasetConfig",
    "GroundTruth",
    "SyntheticDataGenerator",
    "BenchmarkReport",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "PerformanceProfiler",
    "RegressionDetector",
    "BenchmarkScenario",
    "PipelineStep",
    "ScenarioValidator",
    "SimulatedPipeline",
    "StandardBenchmarks",
]
