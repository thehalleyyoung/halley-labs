"""Benchmark suite for the CausalQD framework.

Provides standard Bayesian network benchmarks, synthetic benchmark
generation, and a benchmark runner for systematic evaluation.

Modules
-------
* :mod:`standard_networks` – Asia, Sachs, Insurance, Alarm, Child
* :mod:`synthetic` – random DAG, scalability, faithfulness, sparsity
* :mod:`runner` – benchmark runner and comparison utilities
"""

from causal_qd.benchmarks.standard_networks import (
    AlarmBenchmark,
    AsiaBenchmark,
    ChildBenchmark,
    InsuranceBenchmark,
    SachsBenchmark,
)
from causal_qd.benchmarks.synthetic import (
    FaithfulnessViolationBenchmark,
    RandomDAGBenchmark,
    ScalabilityBenchmark,
    SparsityBenchmark,
)
from causal_qd.benchmarks.runner import BenchmarkRunner, ComparisonRunner

__all__ = [
    "AsiaBenchmark",
    "SachsBenchmark",
    "InsuranceBenchmark",
    "AlarmBenchmark",
    "ChildBenchmark",
    "RandomDAGBenchmark",
    "ScalabilityBenchmark",
    "FaithfulnessViolationBenchmark",
    "SparsityBenchmark",
    "BenchmarkRunner",
    "ComparisonRunner",
]
