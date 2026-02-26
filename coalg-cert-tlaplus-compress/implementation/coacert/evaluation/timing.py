"""
Wall-clock and CPU timing infrastructure for the CoaCert pipeline.

Provides Timer context managers, phase tracking, and statistical
aggregation of timing data across multiple benchmark runs.
"""

from __future__ import annotations

import time
import math
import json
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple,
)


class Phase(Enum):
    """Pipeline phases that can be individually timed."""
    PARSE = auto()
    EXPLORE = auto()
    LEARN = auto()
    COMPRESS = auto()
    VERIFY = auto()
    TOTAL = auto()


@dataclass
class TimingRecord:
    """A single timing measurement."""
    phase: str
    wall_seconds: float
    cpu_seconds: float
    start_wall: float
    end_wall: float
    start_cpu: float
    end_cpu: float
    sub_phases: Dict[str, "TimingRecord"] = field(default_factory=dict)

    @property
    def overhead_fraction(self) -> float:
        """Fraction of wall time not accounted for by sub-phases."""
        if not self.sub_phases or self.wall_seconds <= 0:
            return 0.0
        sub_total = sum(sp.wall_seconds for sp in self.sub_phases.values())
        return max(0.0, 1.0 - sub_total / self.wall_seconds)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "phase": self.phase,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "overhead_fraction": self.overhead_fraction,
        }
        if self.sub_phases:
            d["sub_phases"] = {
                k: v.to_dict() for k, v in self.sub_phases.items()
            }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TimingRecord":
        sub = {}
        if "sub_phases" in d:
            sub = {k: cls.from_dict(v) for k, v in d["sub_phases"].items()}
        return cls(
            phase=d["phase"],
            wall_seconds=d["wall_seconds"],
            cpu_seconds=d["cpu_seconds"],
            start_wall=d.get("start_wall", 0.0),
            end_wall=d.get("end_wall", 0.0),
            start_cpu=d.get("start_cpu", 0.0),
            end_cpu=d.get("end_cpu", 0.0),
            sub_phases=sub,
        )


@dataclass
class TimingStats:
    """Statistical summary over multiple timing observations."""
    phase: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    p25: float
    p75: float
    p95: float
    cpu_mean: float
    cpu_median: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "min": self.min_val,
            "max": self.max_val,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "cpu_mean": self.cpu_mean,
            "cpu_median": self.cpu_median,
        }


def _percentile(data: List[float], pct: float) -> float:
    """Compute the pct-th percentile of sorted data using linear interpolation."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    f = int(math.floor(k))
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def compute_timing_stats(phase: str, records: Sequence[TimingRecord]) -> TimingStats:
    """Aggregate a sequence of TimingRecords into a TimingStats summary."""
    if not records:
        return TimingStats(
            phase=phase, count=0, mean=0, median=0, std_dev=0,
            min_val=0, max_val=0, p25=0, p75=0, p95=0,
            cpu_mean=0, cpu_median=0,
        )
    walls = [r.wall_seconds for r in records]
    cpus = [r.cpu_seconds for r in records]
    n = len(walls)
    mean_w = statistics.mean(walls)
    median_w = statistics.median(walls)
    std_w = statistics.stdev(walls) if n >= 2 else 0.0
    return TimingStats(
        phase=phase,
        count=n,
        mean=mean_w,
        median=median_w,
        std_dev=std_w,
        min_val=min(walls),
        max_val=max(walls),
        p25=_percentile(walls, 25),
        p75=_percentile(walls, 75),
        p95=_percentile(walls, 95),
        cpu_mean=statistics.mean(cpus),
        cpu_median=statistics.median(cpus),
    )


class Timer:
    """Hierarchical timer with context-manager support.

    Usage::

        timer = Timer()
        with timer.phase("parse"):
            with timer.sub_phase("parse", "lex"):
                ...
            with timer.sub_phase("parse", "build_ast"):
                ...
        print(timer.records)
    """

    def __init__(self) -> None:
        self._records: Dict[str, TimingRecord] = {}
        self._stack: List[Tuple[str, float, float]] = []

    @property
    def records(self) -> Dict[str, TimingRecord]:
        return dict(self._records)

    @contextmanager
    def phase(self, name: str):
        """Time a top-level pipeline phase."""
        w0 = time.perf_counter()
        c0 = time.process_time()
        try:
            yield
        finally:
            w1 = time.perf_counter()
            c1 = time.process_time()
            rec = TimingRecord(
                phase=name,
                wall_seconds=w1 - w0,
                cpu_seconds=c1 - c0,
                start_wall=w0, end_wall=w1,
                start_cpu=c0, end_cpu=c1,
            )
            self._records[name] = rec

    @contextmanager
    def sub_phase(self, parent: str, name: str):
        """Time a sub-phase within a parent phase."""
        w0 = time.perf_counter()
        c0 = time.process_time()
        try:
            yield
        finally:
            w1 = time.perf_counter()
            c1 = time.process_time()
            rec = TimingRecord(
                phase=name,
                wall_seconds=w1 - w0,
                cpu_seconds=c1 - c0,
                start_wall=w0, end_wall=w1,
                start_cpu=c0, end_cpu=c1,
            )
            if parent in self._records:
                self._records[parent].sub_phases[name] = rec
            else:
                self._records.setdefault(parent, TimingRecord(
                    phase=parent, wall_seconds=0, cpu_seconds=0,
                    start_wall=0, end_wall=0, start_cpu=0, end_cpu=0,
                ))
                self._records[parent].sub_phases[name] = rec

    def total_wall(self) -> float:
        return sum(r.wall_seconds for r in self._records.values())

    def total_cpu(self) -> float:
        return sum(r.cpu_seconds for r in self._records.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v.to_dict() for k, v in self._records.items()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Timer":
        t = cls()
        for k, v in d.items():
            t._records[k] = TimingRecord.from_dict(v)
        return t


class MultiRunTimer:
    """Collect timing data across multiple runs and compute statistics."""

    def __init__(self) -> None:
        self._runs: List[Timer] = []

    @property
    def run_count(self) -> int:
        return len(self._runs)

    def add_run(self, timer: Timer) -> None:
        self._runs.append(timer)

    def phase_names(self) -> List[str]:
        names: List[str] = []
        seen: set = set()
        for run in self._runs:
            for p in run.records:
                if p not in seen:
                    seen.add(p)
                    names.append(p)
        return names

    def stats_for_phase(self, phase: str) -> TimingStats:
        recs = [
            run.records[phase]
            for run in self._runs
            if phase in run.records
        ]
        return compute_timing_stats(phase, recs)

    def all_stats(self) -> Dict[str, TimingStats]:
        return {p: self.stats_for_phase(p) for p in self.phase_names()}

    def compare(self, other: "MultiRunTimer") -> Dict[str, Dict[str, float]]:
        """Compare this timer against another, returning speedup ratios."""
        result: Dict[str, Dict[str, float]] = {}
        all_phases = set(self.phase_names()) | set(other.phase_names())
        for p in all_phases:
            s1 = self.stats_for_phase(p)
            s2 = other.stats_for_phase(p)
            if s1.mean > 0 and s2.mean > 0:
                result[p] = {
                    "speedup": s2.mean / s1.mean,
                    "self_mean": s1.mean,
                    "other_mean": s2.mean,
                    "self_std": s1.std_dev,
                    "other_std": s2.std_dev,
                }
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_count": self.run_count,
            "runs": [r.to_dict() for r in self._runs],
            "stats": {k: v.to_dict() for k, v in self.all_stats().items()},
        }


def estimate_overhead(timer: Timer) -> Dict[str, float]:
    """Estimate framework overhead per phase.

    Returns a dict mapping phase name to the estimated fraction of
    wall time spent in framework bookkeeping (as opposed to the
    actual computation tracked by sub-phases).
    """
    result: Dict[str, float] = {}
    for name, rec in timer.records.items():
        result[name] = rec.overhead_fraction
    return result


def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 60:
        return f"{seconds:.3f} s"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}m {secs:.1f}s"


def timing_table(stats: Dict[str, TimingStats], title: str = "Timing") -> str:
    """Render a timing-stats dict as a formatted console table."""
    header = f"{'Phase':<20} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'N':>5}"
    sep = "-" * len(header)
    lines = [title, sep, header, sep]
    for phase, s in stats.items():
        lines.append(
            f"{phase:<20} "
            f"{format_duration(s.mean):>10} "
            f"{format_duration(s.median):>10} "
            f"{format_duration(s.std_dev):>10} "
            f"{format_duration(s.min_val):>10} "
            f"{format_duration(s.max_val):>10} "
            f"{s.count:>5}"
        )
    lines.append(sep)
    return "\n".join(lines)
