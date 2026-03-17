"""Evaluation module for benchmarking collusion certification methods."""

try:
    from collusion_proof.evaluation.benchmark_runner import BenchmarkRunner
    from collusion_proof.evaluation.metrics import MetricsCollector
except ImportError:
    pass
