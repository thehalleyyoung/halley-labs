"""
Benchmark runner for the CoaCert pipeline.

Orchestrates full pipeline runs (parse → explore → compress → verify),
collects metrics at each phase, handles warm-up, timeouts, multiple
configurations, statistical aggregation, result comparison, and
regression detection.
"""

from __future__ import annotations

import copy
import json
import math
import os
import signal
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

from .metrics import (
    AggregatedMetrics,
    MetricsCollector,
    PipelineMetrics,
    aggregate_metrics,
    save_metrics_json,
    load_metrics_json,
)
from .timing import Timer, MultiRunTimer, format_duration, timing_table


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class BenchmarkStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    SKIPPED = auto()


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""
    name: str
    spec_path: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    num_runs: int = 5
    warmup_runs: int = 1
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "spec_path": self.spec_path,
            "parameters": self.parameters,
            "timeout_seconds": self.timeout_seconds,
            "num_runs": self.num_runs,
            "warmup_runs": self.warmup_runs,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkConfig":
        return cls(
            name=d["name"],
            spec_path=d.get("spec_path", ""),
            parameters=d.get("parameters", {}),
            timeout_seconds=d.get("timeout_seconds", 300.0),
            num_runs=d.get("num_runs", 5),
            warmup_runs=d.get("warmup_runs", 1),
            tags=d.get("tags", []),
        )


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark (multiple runs)."""
    config: BenchmarkConfig
    status: BenchmarkStatus = BenchmarkStatus.PENDING
    runs: List[PipelineMetrics] = field(default_factory=list)
    warmup_runs_data: List[PipelineMetrics] = field(default_factory=list)
    aggregated: Optional[AggregatedMetrics] = None
    error_message: str = ""
    elapsed_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "status": self.status.name,
            "runs": [r.to_dict() for r in self.runs],
            "warmup_runs": len(self.warmup_runs_data),
            "aggregated": self.aggregated.to_dict() if self.aggregated else None,
            "error_message": self.error_message,
            "elapsed_total": self.elapsed_total,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        cfg = BenchmarkConfig.from_dict(d["config"])
        runs = [PipelineMetrics.from_dict(r) for r in d.get("runs", [])]
        agg = None
        if d.get("aggregated"):
            # reconstruct lightly
            agg = aggregate_metrics(runs) if runs else None
        return cls(
            config=cfg,
            status=BenchmarkStatus[d.get("status", "PENDING")],
            runs=runs,
            aggregated=agg,
            error_message=d.get("error_message", ""),
            elapsed_total=d.get("elapsed_total", 0.0),
        )


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results across a full benchmark suite."""
    results: List[BenchmarkResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def elapsed(self) -> float:
        return self.finished_at - self.started_at

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.results if r.status == BenchmarkStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results
                   if r.status in (BenchmarkStatus.FAILED, BenchmarkStatus.TIMEOUT))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "elapsed": self.elapsed,
            "completed": self.completed_count,
            "failed": self.failed_count,
        }


# ---------------------------------------------------------------------------
# Timeout support
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


@contextmanager
def _timeout_context(seconds: float):
    """Cross-platform timeout; uses SIGALRM on Unix, threading on others."""
    if seconds <= 0:
        yield
        return

    if hasattr(signal, "SIGALRM"):
        def _handler(signum, frame):
            raise TimeoutError(f"Benchmark timed out after {seconds:.1f}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    else:
        import threading
        timer_fired = threading.Event()

        def _fire():
            timer_fired.set()

        t = threading.Timer(seconds, _fire)
        t.daemon = True
        t.start()
        try:
            yield
        finally:
            t.cancel()
            if timer_fired.is_set():
                raise TimeoutError(f"Benchmark timed out after {seconds:.1f}s")


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

class ProgressReporter:
    """Simple console progress reporter."""

    def __init__(self, total: int, verbose: bool = True) -> None:
        self._total = total
        self._current = 0
        self._verbose = verbose
        self._start = time.monotonic()

    def start_benchmark(self, name: str, run_idx: int, total_runs: int) -> None:
        if not self._verbose:
            return
        pct = (self._current / self._total * 100) if self._total else 0
        print(
            f"  [{self._current + 1}/{self._total}] "
            f"{name} run {run_idx + 1}/{total_runs}  ({pct:.0f}%)",
            flush=True,
        )

    def finish_benchmark(self, name: str, elapsed: float) -> None:
        self._current += 1
        if not self._verbose:
            return
        print(
            f"    ✓ {name} completed in {format_duration(elapsed)}",
            flush=True,
        )

    def fail_benchmark(self, name: str, reason: str) -> None:
        self._current += 1
        if not self._verbose:
            return
        print(f"    ✗ {name} FAILED: {reason}", flush=True)

    def finish_all(self) -> None:
        if not self._verbose:
            return
        total_time = time.monotonic() - self._start
        print(
            f"\nAll benchmarks done in {format_duration(total_time)} "
            f"({self._current}/{self._total} completed)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Pipeline phase runners (pluggable)
# ---------------------------------------------------------------------------

@dataclass
class PipelinePhases:
    """Pluggable pipeline phases.

    Each phase is a callable ``(config, context) -> context`` where
    ``context`` is a mutable dict carrying intermediate state.
    """
    parse: Optional[Callable[[BenchmarkConfig, Dict], Dict]] = None
    explore: Optional[Callable[[BenchmarkConfig, Dict], Dict]] = None
    learn: Optional[Callable[[BenchmarkConfig, Dict], Dict]] = None
    compress: Optional[Callable[[BenchmarkConfig, Dict], Dict]] = None
    verify: Optional[Callable[[BenchmarkConfig, Dict], Dict]] = None


def _default_parse(cfg: BenchmarkConfig, ctx: Dict) -> Dict:
    """Default parse stub – reads spec file if it exists."""
    path = cfg.spec_path
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            ctx["spec_source"] = f.read()
        ctx["spec_lines"] = ctx["spec_source"].count("\n") + 1
    return ctx


def _noop_phase(cfg: BenchmarkConfig, ctx: Dict) -> Dict:
    return ctx


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run benchmarks with timing, metrics collection, and aggregation.

    Parameters
    ----------
    phases : PipelinePhases
        Pluggable callables for each pipeline phase.
    verbose : bool
        Whether to print progress.
    output_dir : str or None
        Directory to write result JSON files.
    """

    def __init__(
        self,
        phases: Optional[PipelinePhases] = None,
        verbose: bool = True,
        output_dir: Optional[str] = None,
    ) -> None:
        p = phases or PipelinePhases()
        self._phases = {
            "parse": p.parse or _default_parse,
            "explore": p.explore or _noop_phase,
            "learn": p.learn or _noop_phase,
            "compress": p.compress or _noop_phase,
            "verify": p.verify or _noop_phase,
        }
        self._verbose = verbose
        self._output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # -- single run ----------------------------------------------------------

    def _run_once(
        self,
        config: BenchmarkConfig,
        run_index: int,
    ) -> PipelineMetrics:
        """Execute one complete pipeline run and collect metrics."""
        timer = Timer()
        collector = MetricsCollector(config.name, run_index)
        ctx: Dict[str, Any] = {"config": config, "collector": collector}

        phase_order = ["parse", "explore", "learn", "compress", "verify"]
        for phase_name in phase_order:
            fn = self._phases[phase_name]
            with timer.phase(phase_name):
                ctx = fn(config, ctx)

        collector.record_timings_from_timer(timer)

        # Extract sizes from context if phases populated them
        if "original_states" in ctx:
            collector.record_original_size(
                states=ctx["original_states"],
                transitions=ctx.get("original_transitions", 0),
                actions=ctx.get("original_actions", 0),
            )
        if "quotient_states" in ctx:
            collector.record_quotient_size(
                states=ctx["quotient_states"],
                transitions=ctx.get("quotient_transitions", 0),
                actions=ctx.get("quotient_actions", 0),
            )
        if "membership_queries" in ctx:
            collector.record_queries(
                membership=ctx.get("membership_queries", 0),
                equivalence=ctx.get("equivalence_queries", 0),
                learning_rounds=ctx.get("learning_rounds", 0),
            )
        if "witness_size_bytes" in ctx:
            collector.record_witness(
                size_bytes=ctx.get("witness_size_bytes", 0),
                partition_blocks=ctx.get("partition_blocks", 0),
                morphism_edges=ctx.get("morphism_edges", 0),
                obligations=ctx.get("proof_obligations", 0),
                discharged=ctx.get("proof_obligations_discharged", 0),
            )

        return collector.finalize()

    # -- benchmark execution -------------------------------------------------

    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark with warmup, multiple runs, and aggregation."""
        result = BenchmarkResult(config=config, status=BenchmarkStatus.RUNNING)
        t0 = time.monotonic()
        total_individual = config.warmup_runs + config.num_runs

        progress = ProgressReporter(total_individual, self._verbose)

        # Warmup runs
        for i in range(config.warmup_runs):
            progress.start_benchmark(config.name, i, config.warmup_runs)
            try:
                with _timeout_context(config.timeout_seconds):
                    m = self._run_once(config, run_index=-1 - i)
                result.warmup_runs_data.append(m)
                progress.finish_benchmark(
                    f"{config.name} (warmup)", sum(m.phase_timings.values())
                )
            except TimeoutError as e:
                progress.fail_benchmark(config.name, str(e))
                result.status = BenchmarkStatus.TIMEOUT
                result.error_message = str(e)
                result.elapsed_total = time.monotonic() - t0
                return result
            except Exception as e:
                progress.fail_benchmark(config.name, str(e))
                # warmup failure is not fatal; continue

        # Measured runs
        for i in range(config.num_runs):
            progress.start_benchmark(config.name, i, config.num_runs)
            try:
                with _timeout_context(config.timeout_seconds):
                    m = self._run_once(config, run_index=i)
                result.runs.append(m)
                progress.finish_benchmark(
                    config.name, sum(m.phase_timings.values())
                )
            except TimeoutError as e:
                progress.fail_benchmark(config.name, str(e))
                result.status = BenchmarkStatus.TIMEOUT
                result.error_message = str(e)
                break
            except Exception as e:
                progress.fail_benchmark(config.name, str(e))
                result.status = BenchmarkStatus.FAILED
                result.error_message = f"{type(e).__name__}: {e}"
                break

        if result.status == BenchmarkStatus.RUNNING:
            result.status = BenchmarkStatus.COMPLETED

        if result.runs:
            result.aggregated = aggregate_metrics(result.runs)

        result.elapsed_total = time.monotonic() - t0
        return result

    # -- suite execution -----------------------------------------------------

    def run_suite(
        self,
        configs: Sequence[BenchmarkConfig],
    ) -> BenchmarkSuiteResult:
        """Run a suite of benchmarks sequentially."""
        suite = BenchmarkSuiteResult(started_at=time.monotonic())
        if self._verbose:
            print(f"\n{'='*60}")
            print(f"  Running {len(configs)} benchmarks")
            print(f"{'='*60}\n")

        for cfg in configs:
            br = self.run_benchmark(cfg)
            suite.results.append(br)

        suite.finished_at = time.monotonic()

        if self._verbose:
            c = suite.completed_count
            f = suite.failed_count
            print(f"\nSuite finished: {c} completed, {f} failed "
                  f"in {format_duration(suite.elapsed)}")

        if self._output_dir:
            path = os.path.join(self._output_dir, "suite_results.json")
            with open(path, "w") as fh:
                json.dump(suite.to_dict(), fh, indent=2)

        return suite

    # -- result persistence --------------------------------------------------

    def save_result(self, result: BenchmarkResult, path: str) -> None:
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def load_result(self, path: str) -> BenchmarkResult:
        with open(path, "r") as f:
            return BenchmarkResult.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Comparison and regression detection
# ---------------------------------------------------------------------------

@dataclass
class RegressionInfo:
    """Information about a detected performance regression."""
    benchmark: str
    metric: str
    baseline_value: float
    current_value: float
    change_pct: float
    threshold_pct: float
    is_regression: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "metric": self.metric,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "change_pct": self.change_pct,
            "threshold_pct": self.threshold_pct,
            "is_regression": self.is_regression,
        }


def compare_results(
    baseline: BenchmarkSuiteResult,
    current: BenchmarkSuiteResult,
) -> Dict[str, Dict[str, Any]]:
    """Compare two suite results, returning per-benchmark deltas."""
    b_map = {r.config.name: r for r in baseline.results}
    c_map = {r.config.name: r for r in current.results}
    comparison: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(b_map.keys()) | set(c_map.keys())):
        br = b_map.get(name)
        cr = c_map.get(name)
        entry: Dict[str, Any] = {"baseline_status": None, "current_status": None}
        if br:
            entry["baseline_status"] = br.status.name
        if cr:
            entry["current_status"] = cr.status.name
        if br and cr and br.aggregated and cr.aggregated:
            b_agg = br.aggregated
            c_agg = cr.aggregated
            if b_agg.mean_total_time > 0:
                entry["time_change_pct"] = (
                    (c_agg.mean_total_time - b_agg.mean_total_time)
                    / b_agg.mean_total_time * 100
                )
            entry["baseline_time"] = b_agg.mean_total_time
            entry["current_time"] = c_agg.mean_total_time
            entry["baseline_compression"] = b_agg.state_compression_ratio
            entry["current_compression"] = c_agg.state_compression_ratio
            if b_agg.state_compression_ratio > 0:
                entry["compression_change_pct"] = (
                    (c_agg.state_compression_ratio - b_agg.state_compression_ratio)
                    / b_agg.state_compression_ratio * 100
                )
        comparison[name] = entry
    return comparison


def detect_regressions(
    baseline: BenchmarkSuiteResult,
    current: BenchmarkSuiteResult,
    time_threshold_pct: float = 10.0,
    compression_threshold_pct: float = 5.0,
) -> List[RegressionInfo]:
    """Detect regressions relative to a baseline.

    A regression is flagged if the current run is worse than baseline
    by more than the given threshold percentage.

    Parameters
    ----------
    time_threshold_pct : float
        Percentage increase in time that constitutes a regression.
    compression_threshold_pct : float
        Percentage worsening (increase) in compression ratio that
        constitutes a regression.
    """
    regressions: List[RegressionInfo] = []
    comp = compare_results(baseline, current)

    for name, entry in comp.items():
        # Time regression
        if "time_change_pct" in entry:
            change = entry["time_change_pct"]
            regressions.append(RegressionInfo(
                benchmark=name,
                metric="mean_total_time",
                baseline_value=entry["baseline_time"],
                current_value=entry["current_time"],
                change_pct=change,
                threshold_pct=time_threshold_pct,
                is_regression=change > time_threshold_pct,
            ))

        # Compression regression (higher ratio = worse)
        if "compression_change_pct" in entry:
            change = entry["compression_change_pct"]
            regressions.append(RegressionInfo(
                benchmark=name,
                metric="state_compression_ratio",
                baseline_value=entry["baseline_compression"],
                current_value=entry["current_compression"],
                change_pct=change,
                threshold_pct=compression_threshold_pct,
                is_regression=change > compression_threshold_pct,
            ))

    return regressions


def regression_report(regressions: Sequence[RegressionInfo]) -> str:
    """Format regression info as a console report."""
    if not regressions:
        return "No metrics to compare."
    flagged = [r for r in regressions if r.is_regression]
    lines: List[str] = []
    lines.append(f"Regression Analysis ({len(regressions)} metrics checked)")
    lines.append("-" * 70)

    hdr = (
        f"{'Benchmark':<24} {'Metric':<25} "
        f"{'Baseline':>10} {'Current':>10} {'Change':>8} {'Flag':>6}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in regressions:
        flag = " ⚠ " if r.is_regression else " ✓ "
        lines.append(
            f"{r.benchmark:<24} {r.metric:<25} "
            f"{r.baseline_value:>10.4f} {r.current_value:>10.4f} "
            f"{r.change_pct:>+7.1f}% {flag:>6}"
        )

    lines.append("-" * len(hdr))
    if flagged:
        lines.append(f"\n⚠ {len(flagged)} regression(s) detected!")
    else:
        lines.append("\n✓ No regressions detected.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: load suite from JSON
# ---------------------------------------------------------------------------

def load_suite_result(path: str) -> BenchmarkSuiteResult:
    with open(path, "r") as f:
        data = json.load(f)
    suite = BenchmarkSuiteResult()
    suite.started_at = data.get("started_at", 0)
    suite.finished_at = data.get("finished_at", 0)
    for rd in data.get("results", []):
        suite.results.append(BenchmarkResult.from_dict(rd))
    return suite


def save_suite_result(suite: BenchmarkSuiteResult, path: str) -> None:
    with open(path, "w") as f:
        json.dump(suite.to_dict(), f, indent=2)
