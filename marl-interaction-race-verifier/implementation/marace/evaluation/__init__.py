"""MARACE evaluation module.

Provides benchmarks with planted races, metric collection,
baseline comparisons, and experiment management for evaluating
the MARACE verification framework.
"""

from .benchmarks import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkSuite,
    HighwayIntersectionBenchmark,
    HighwayMergingBenchmark,
    WarehouseCorridorBenchmark,
    TradingBenchmark,
    PlantedRace,
    ScalabilityBenchmark,
)
from .metrics import (
    MetricCollector,
    DetectionRecall,
    FalsePositiveRate,
    SoundCoverage,
    TimeToDetection,
    ProbabilityBoundAccuracy,
    ScalabilityMetric,
    MetricsAggregator,
    MetricsFormatter,
)
from .baselines import (
    BaselineRunner,
    SingleAgentVerifier,
    BruteForceSimulator,
    NaiveEnumerator,
    AblationRunner,
    AblationConfig,
    BaselineComparator,
)
from .experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    ResultDatabase,
    ExperimentComparator,
    ReproducibilityChecker,
    ExperimentSweep,
    ProgressTracker,
)

__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkSuite",
    "HighwayIntersectionBenchmark",
    "HighwayMergingBenchmark",
    "WarehouseCorridorBenchmark",
    "TradingBenchmark",
    "PlantedRace",
    "ScalabilityBenchmark",
    "MetricCollector",
    "DetectionRecall",
    "FalsePositiveRate",
    "SoundCoverage",
    "TimeToDetection",
    "ProbabilityBoundAccuracy",
    "ScalabilityMetric",
    "MetricsAggregator",
    "MetricsFormatter",
    "BaselineRunner",
    "SingleAgentVerifier",
    "BruteForceSimulator",
    "NaiveEnumerator",
    "AblationRunner",
    "AblationConfig",
    "BaselineComparator",
    "Experiment",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "ResultDatabase",
    "ExperimentComparator",
    "ReproducibilityChecker",
    "ExperimentSweep",
    "ProgressTracker",
]
