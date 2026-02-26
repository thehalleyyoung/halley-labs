"""
Metrics collection and aggregation for the CoaCert evaluation harness.

Tracks compression ratios, state/transition counts, query counts,
timing, memory usage, and throughput across pipeline runs.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union,
)

from .timing import Timer, TimingRecord, TimingStats, compute_timing_stats


class MetricKind(Enum):
    """Categories of metrics we collect."""
    STATE_SPACE = auto()
    TRANSITIONS = auto()
    COMPRESSION = auto()
    TIMING = auto()
    MEMORY = auto()
    QUERIES = auto()
    LEARNING = auto()
    WITNESS = auto()
    THROUGHPUT = auto()


@dataclass
class StateSpaceMetrics:
    """Sizes of the original and quotient state spaces."""
    original_states: int = 0
    quotient_states: int = 0
    original_transitions: int = 0
    quotient_transitions: int = 0
    original_actions: int = 0
    quotient_actions: int = 0

    @property
    def state_compression_ratio(self) -> float:
        if self.original_states == 0:
            return 1.0
        return self.quotient_states / self.original_states

    @property
    def transition_compression_ratio(self) -> float:
        if self.original_transitions == 0:
            return 1.0
        return self.quotient_transitions / self.original_transitions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_states": self.original_states,
            "quotient_states": self.quotient_states,
            "original_transitions": self.original_transitions,
            "quotient_transitions": self.quotient_transitions,
            "original_actions": self.original_actions,
            "quotient_actions": self.quotient_actions,
            "state_compression_ratio": self.state_compression_ratio,
            "transition_compression_ratio": self.transition_compression_ratio,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StateSpaceMetrics":
        return cls(
            original_states=d.get("original_states", 0),
            quotient_states=d.get("quotient_states", 0),
            original_transitions=d.get("original_transitions", 0),
            quotient_transitions=d.get("quotient_transitions", 0),
            original_actions=d.get("original_actions", 0),
            quotient_actions=d.get("quotient_actions", 0),
        )


@dataclass
class MemoryMetrics:
    """Peak and current memory usage (bytes)."""
    peak_rss_bytes: int = 0
    current_rss_bytes: int = 0
    peak_vms_bytes: int = 0

    @property
    def peak_rss_mb(self) -> float:
        return self.peak_rss_bytes / (1024 * 1024)

    @property
    def current_rss_mb(self) -> float:
        return self.current_rss_bytes / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peak_rss_bytes": self.peak_rss_bytes,
            "current_rss_bytes": self.current_rss_bytes,
            "peak_vms_bytes": self.peak_vms_bytes,
            "peak_rss_mb": self.peak_rss_mb,
            "current_rss_mb": self.current_rss_mb,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryMetrics":
        return cls(
            peak_rss_bytes=d.get("peak_rss_bytes", 0),
            current_rss_bytes=d.get("current_rss_bytes", 0),
            peak_vms_bytes=d.get("peak_vms_bytes", 0),
        )


@dataclass
class QueryMetrics:
    """Counts of membership and equivalence queries during learning."""
    membership_queries: int = 0
    equivalence_queries: int = 0
    membership_cache_hits: int = 0
    equivalence_cache_hits: int = 0
    learning_rounds: int = 0

    @property
    def total_queries(self) -> int:
        return self.membership_queries + self.equivalence_queries

    @property
    def cache_hit_rate(self) -> float:
        total = self.membership_queries + self.equivalence_queries
        if total == 0:
            return 0.0
        hits = self.membership_cache_hits + self.equivalence_cache_hits
        return hits / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "membership_queries": self.membership_queries,
            "equivalence_queries": self.equivalence_queries,
            "membership_cache_hits": self.membership_cache_hits,
            "equivalence_cache_hits": self.equivalence_cache_hits,
            "learning_rounds": self.learning_rounds,
            "total_queries": self.total_queries,
            "cache_hit_rate": self.cache_hit_rate,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QueryMetrics":
        return cls(
            membership_queries=d.get("membership_queries", 0),
            equivalence_queries=d.get("equivalence_queries", 0),
            membership_cache_hits=d.get("membership_cache_hits", 0),
            equivalence_cache_hits=d.get("equivalence_cache_hits", 0),
            learning_rounds=d.get("learning_rounds", 0),
        )


@dataclass
class WitnessMetrics:
    """Size and structure of the generated witness / certificate."""
    witness_size_bytes: int = 0
    partition_block_count: int = 0
    morphism_edge_count: int = 0
    proof_obligation_count: int = 0
    proof_obligations_discharged: int = 0

    @property
    def witness_size_kb(self) -> float:
        return self.witness_size_bytes / 1024.0

    @property
    def discharge_rate(self) -> float:
        if self.proof_obligation_count == 0:
            return 1.0
        return self.proof_obligations_discharged / self.proof_obligation_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "witness_size_bytes": self.witness_size_bytes,
            "witness_size_kb": self.witness_size_kb,
            "partition_block_count": self.partition_block_count,
            "morphism_edge_count": self.morphism_edge_count,
            "proof_obligation_count": self.proof_obligation_count,
            "proof_obligations_discharged": self.proof_obligations_discharged,
            "discharge_rate": self.discharge_rate,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WitnessMetrics":
        return cls(
            witness_size_bytes=d.get("witness_size_bytes", 0),
            partition_block_count=d.get("partition_block_count", 0),
            morphism_edge_count=d.get("morphism_edge_count", 0),
            proof_obligation_count=d.get("proof_obligation_count", 0),
            proof_obligations_discharged=d.get("proof_obligations_discharged", 0),
        )


@dataclass
class ThroughputMetrics:
    """Derived throughput numbers."""
    states_per_second: float = 0.0
    transitions_per_second: float = 0.0
    queries_per_second: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "states_per_second": self.states_per_second,
            "transitions_per_second": self.transitions_per_second,
            "queries_per_second": self.queries_per_second,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThroughputMetrics":
        return cls(
            states_per_second=d.get("states_per_second", 0.0),
            transitions_per_second=d.get("transitions_per_second", 0.0),
            queries_per_second=d.get("queries_per_second", 0.0),
        )


@dataclass
class PipelineMetrics:
    """Complete metrics for a single pipeline run."""
    benchmark_name: str = ""
    run_index: int = 0
    state_space: StateSpaceMetrics = field(default_factory=StateSpaceMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    queries: QueryMetrics = field(default_factory=QueryMetrics)
    witness: WitnessMetrics = field(default_factory=WitnessMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    phase_timings: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "run_index": self.run_index,
            "state_space": self.state_space.to_dict(),
            "memory": self.memory.to_dict(),
            "queries": self.queries.to_dict(),
            "witness": self.witness.to_dict(),
            "throughput": self.throughput.to_dict(),
            "phase_timings": self.phase_timings,
            "timestamp": self.timestamp,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineMetrics":
        return cls(
            benchmark_name=d.get("benchmark_name", ""),
            run_index=d.get("run_index", 0),
            state_space=StateSpaceMetrics.from_dict(d.get("state_space", {})),
            memory=MemoryMetrics.from_dict(d.get("memory", {})),
            queries=QueryMetrics.from_dict(d.get("queries", {})),
            witness=WitnessMetrics.from_dict(d.get("witness", {})),
            throughput=ThroughputMetrics.from_dict(d.get("throughput", {})),
            phase_timings=d.get("phase_timings", {}),
            timestamp=d.get("timestamp", 0),
            extra=d.get("extra", {}),
        )


def _try_get_memory() -> MemoryMetrics:
    """Best-effort memory sampling via resource module."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        import sys
        factor = 1024 if sys.platform == "linux" else 1
        peak = int(ru.ru_maxrss * factor)
        return MemoryMetrics(peak_rss_bytes=peak, current_rss_bytes=peak)
    except Exception:
        return MemoryMetrics()


class MetricsCollector:
    """Collects metrics throughout a pipeline run.

    Usage::

        mc = MetricsCollector("my_benchmark")
        mc.record_original_size(states=100, transitions=300, actions=3)
        mc.record_quotient_size(states=20, transitions=60, actions=3)
        mc.record_queries(membership=450, equivalence=12)
        mc.record_timing("parse", 0.05)
        mc.record_timing("explore", 1.2)
        mc.finalize()
        print(mc.current_metrics.to_dict())
    """

    def __init__(self, benchmark_name: str = "", run_index: int = 0) -> None:
        self._metrics = PipelineMetrics(
            benchmark_name=benchmark_name,
            run_index=run_index,
        )
        self._finalized = False

    @property
    def current_metrics(self) -> PipelineMetrics:
        return self._metrics

    # -- state space ----------------------------------------------------------

    def record_original_size(
        self, states: int, transitions: int, actions: int = 0
    ) -> None:
        self._metrics.state_space.original_states = states
        self._metrics.state_space.original_transitions = transitions
        self._metrics.state_space.original_actions = actions

    def record_quotient_size(
        self, states: int, transitions: int, actions: int = 0
    ) -> None:
        self._metrics.state_space.quotient_states = states
        self._metrics.state_space.quotient_transitions = transitions
        self._metrics.state_space.quotient_actions = actions

    # -- queries --------------------------------------------------------------

    def record_queries(
        self,
        membership: int = 0,
        equivalence: int = 0,
        membership_cache_hits: int = 0,
        equivalence_cache_hits: int = 0,
        learning_rounds: int = 0,
    ) -> None:
        q = self._metrics.queries
        q.membership_queries += membership
        q.equivalence_queries += equivalence
        q.membership_cache_hits += membership_cache_hits
        q.equivalence_cache_hits += equivalence_cache_hits
        q.learning_rounds += learning_rounds

    def increment_membership_query(self, cache_hit: bool = False) -> None:
        self._metrics.queries.membership_queries += 1
        if cache_hit:
            self._metrics.queries.membership_cache_hits += 1

    def increment_equivalence_query(self, cache_hit: bool = False) -> None:
        self._metrics.queries.equivalence_queries += 1
        if cache_hit:
            self._metrics.queries.equivalence_cache_hits += 1

    def increment_learning_round(self) -> None:
        self._metrics.queries.learning_rounds += 1

    # -- witness --------------------------------------------------------------

    def record_witness(
        self,
        size_bytes: int = 0,
        partition_blocks: int = 0,
        morphism_edges: int = 0,
        obligations: int = 0,
        discharged: int = 0,
    ) -> None:
        w = self._metrics.witness
        w.witness_size_bytes = size_bytes
        w.partition_block_count = partition_blocks
        w.morphism_edge_count = morphism_edges
        w.proof_obligation_count = obligations
        w.proof_obligations_discharged = discharged

    # -- timing ---------------------------------------------------------------

    def record_timing(self, phase: str, seconds: float) -> None:
        self._metrics.phase_timings[phase] = seconds

    def record_timings_from_timer(self, timer: Timer) -> None:
        for name, rec in timer.records.items():
            self._metrics.phase_timings[name] = rec.wall_seconds

    # -- memory ---------------------------------------------------------------

    def sample_memory(self) -> None:
        self._metrics.memory = _try_get_memory()

    def record_memory(self, peak_rss: int, current_rss: int = 0, peak_vms: int = 0) -> None:
        self._metrics.memory = MemoryMetrics(
            peak_rss_bytes=peak_rss,
            current_rss_bytes=current_rss or peak_rss,
            peak_vms_bytes=peak_vms,
        )

    # -- extra ----------------------------------------------------------------

    def record_extra(self, key: str, value: Any) -> None:
        self._metrics.extra[key] = value

    # -- finalize -------------------------------------------------------------

    def finalize(self) -> PipelineMetrics:
        """Compute derived metrics and return the final PipelineMetrics."""
        if self._finalized:
            return self._metrics
        m = self._metrics
        total_time = sum(m.phase_timings.values())
        if total_time > 0:
            m.throughput.states_per_second = (
                m.state_space.original_states / total_time
            )
            m.throughput.transitions_per_second = (
                m.state_space.original_transitions / total_time
            )
            m.throughput.queries_per_second = (
                m.queries.total_queries / total_time
            )
        self.sample_memory()
        self._finalized = True
        return m


@dataclass
class AggregatedMetrics:
    """Statistical summary across multiple runs of the same benchmark."""
    benchmark_name: str
    run_count: int
    state_compression_ratio: float
    transition_compression_ratio: float
    mean_total_time: float
    median_total_time: float
    std_total_time: float
    mean_queries: float
    mean_memory_mb: float
    per_phase_mean: Dict[str, float] = field(default_factory=dict)
    per_phase_std: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "run_count": self.run_count,
            "state_compression_ratio": self.state_compression_ratio,
            "transition_compression_ratio": self.transition_compression_ratio,
            "mean_total_time": self.mean_total_time,
            "median_total_time": self.median_total_time,
            "std_total_time": self.std_total_time,
            "mean_queries": self.mean_queries,
            "mean_memory_mb": self.mean_memory_mb,
            "per_phase_mean": self.per_phase_mean,
            "per_phase_std": self.per_phase_std,
        }


def aggregate_metrics(runs: Sequence[PipelineMetrics]) -> AggregatedMetrics:
    """Compute summary statistics over multiple PipelineMetrics."""
    if not runs:
        return AggregatedMetrics(
            benchmark_name="", run_count=0,
            state_compression_ratio=0, transition_compression_ratio=0,
            mean_total_time=0, median_total_time=0, std_total_time=0,
            mean_queries=0, mean_memory_mb=0,
        )
    name = runs[0].benchmark_name
    n = len(runs)
    totals = [sum(r.phase_timings.values()) for r in runs]
    mean_t = statistics.mean(totals)
    median_t = statistics.median(totals)
    std_t = statistics.stdev(totals) if n >= 2 else 0.0

    all_phases: set = set()
    for r in runs:
        all_phases |= set(r.phase_timings.keys())
    per_phase_mean: Dict[str, float] = {}
    per_phase_std: Dict[str, float] = {}
    for p in sorted(all_phases):
        vals = [r.phase_timings.get(p, 0.0) for r in runs]
        per_phase_mean[p] = statistics.mean(vals)
        per_phase_std[p] = statistics.stdev(vals) if n >= 2 else 0.0

    cr_states = [r.state_space.state_compression_ratio for r in runs]
    cr_trans = [r.state_space.transition_compression_ratio for r in runs]
    queries = [float(r.queries.total_queries) for r in runs]
    mem = [r.memory.peak_rss_mb for r in runs]

    return AggregatedMetrics(
        benchmark_name=name,
        run_count=n,
        state_compression_ratio=statistics.mean(cr_states),
        transition_compression_ratio=statistics.mean(cr_trans),
        mean_total_time=mean_t,
        median_total_time=median_t,
        std_total_time=std_t,
        mean_queries=statistics.mean(queries),
        mean_memory_mb=statistics.mean(mem),
        per_phase_mean=per_phase_mean,
        per_phase_std=per_phase_std,
    )


def metrics_to_json(metrics: Union[PipelineMetrics, AggregatedMetrics],
                     indent: int = 2) -> str:
    return json.dumps(metrics.to_dict(), indent=indent)


def metrics_to_csv_row(m: PipelineMetrics) -> Dict[str, Any]:
    """Flatten a PipelineMetrics into a single dict suitable for csv.DictWriter."""
    flat: Dict[str, Any] = {
        "benchmark": m.benchmark_name,
        "run_index": m.run_index,
        "original_states": m.state_space.original_states,
        "quotient_states": m.state_space.quotient_states,
        "state_compression": m.state_space.state_compression_ratio,
        "original_transitions": m.state_space.original_transitions,
        "quotient_transitions": m.state_space.quotient_transitions,
        "transition_compression": m.state_space.transition_compression_ratio,
        "membership_queries": m.queries.membership_queries,
        "equivalence_queries": m.queries.equivalence_queries,
        "learning_rounds": m.queries.learning_rounds,
        "witness_size_bytes": m.witness.witness_size_bytes,
        "peak_rss_mb": m.memory.peak_rss_mb,
        "states_per_second": m.throughput.states_per_second,
    }
    for phase, t in m.phase_timings.items():
        flat[f"time_{phase}"] = t
    flat["time_total"] = sum(m.phase_timings.values())
    return flat


def write_csv(runs: Sequence[PipelineMetrics], out: TextIO) -> None:
    """Write a sequence of PipelineMetrics to a CSV file."""
    if not runs:
        return
    rows = [metrics_to_csv_row(r) for r in runs]
    all_keys: List[str] = []
    seen: set = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    writer = csv.DictWriter(out, fieldnames=all_keys)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def load_metrics_json(path: str) -> List[PipelineMetrics]:
    """Load a list of PipelineMetrics from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [PipelineMetrics.from_dict(data)]
    return [PipelineMetrics.from_dict(d) for d in data]


def save_metrics_json(
    metrics: Sequence[PipelineMetrics], path: str, indent: int = 2
) -> None:
    """Save a list of PipelineMetrics to a JSON file."""
    with open(path, "w") as f:
        json.dump([m.to_dict() for m in metrics], f, indent=indent)
