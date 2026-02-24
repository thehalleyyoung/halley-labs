"""
Concurrent performance profiler.

Profile scalability, detect false sharing, lock contention, cache behavior,
and estimate parallel speedup limits using Amdahl's and Gustafson's laws.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import re
import subprocess
import time
import os
import math


class ScalabilityRating(Enum):
    EXCELLENT = "excellent"   # > 0.8 efficiency at max threads
    GOOD = "good"             # 0.5-0.8
    FAIR = "fair"             # 0.3-0.5
    POOR = "poor"             # < 0.3


class ContentionLevel(Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class TimingResult:
    threads: int
    wall_time_seconds: float
    cpu_time_seconds: float = 0.0
    throughput: float = 0.0

    def __str__(self) -> str:
        return f"{self.threads} thread(s): {self.wall_time_seconds:.4f}s (throughput: {self.throughput:.2f})"


@dataclass
class ScalabilityResult:
    timings: List[TimingResult] = field(default_factory=list)
    speedups: Dict[int, float] = field(default_factory=dict)
    efficiencies: Dict[int, float] = field(default_factory=dict)
    rating: ScalabilityRating = ScalabilityRating.FAIR
    bottleneck: str = ""

    def __str__(self) -> str:
        lines = [f"Scalability: [{self.rating.value}]"]
        for t in self.timings:
            sp = self.speedups.get(t.threads, 0)
            eff = self.efficiencies.get(t.threads, 0)
            lines.append(f"  {t.threads} threads: {t.wall_time_seconds:.4f}s "
                         f"(speedup={sp:.2f}x, efficiency={eff:.1%})")
        if self.bottleneck:
            lines.append(f"  Bottleneck: {self.bottleneck}")
        return "\n".join(lines)


@dataclass
class FalseSharing:
    variable_a: str
    variable_b: str
    cache_line_offset: int = 0
    estimated_penalty_pct: float = 0.0
    line: int = 0

    def __str__(self) -> str:
        return (f"False sharing at line {self.line}: '{self.variable_a}' and "
                f"'{self.variable_b}' (penalty ~{self.estimated_penalty_pct:.1f}%)")


@dataclass
class LockContention:
    lock_name: str
    contention_level: ContentionLevel
    wait_time_pct: float = 0.0
    acquire_count: int = 0
    hold_time_avg_us: float = 0.0
    line: int = 0

    def __str__(self) -> str:
        return (f"Lock '{self.lock_name}' [{self.contention_level.value}]: "
                f"wait={self.wait_time_pct:.1f}%, holds={self.acquire_count}, "
                f"avg hold={self.hold_time_avg_us:.1f}µs")


@dataclass
class CacheReport:
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    l3_hit_rate: float = 0.0
    cache_line_bounces: int = 0
    estimated_penalty_cycles: int = 0
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["Cache behavior:"]
        lines.append(f"  L1 hit rate: {self.l1_hit_rate:.1%}")
        lines.append(f"  L2 hit rate: {self.l2_hit_rate:.1%}")
        lines.append(f"  L3 hit rate: {self.l3_hit_rate:.1%}")
        if self.cache_line_bounces:
            lines.append(f"  Cache line bounces: {self.cache_line_bounces}")
        for r in self.recommendations:
            lines.append(f"  * {r}")
        return "\n".join(lines)


@dataclass
class AmdahlResult:
    serial_fraction: float
    max_speedup: float
    speedup_at: Dict[int, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Amdahl's Law: serial fraction = {self.serial_fraction:.2%}"]
        lines.append(f"  Max theoretical speedup: {self.max_speedup:.2f}x")
        for n, sp in sorted(self.speedup_at.items()):
            lines.append(f"  {n} threads: {sp:.2f}x")
        for r in self.recommendations:
            lines.append(f"  * {r}")
        return "\n".join(lines)


@dataclass
class GustafsonResult:
    serial_fraction: float
    scaled_speedups: Dict[int, float] = field(default_factory=dict)
    problem_size_scaling: Dict[int, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Gustafson's Law: serial fraction = {self.serial_fraction:.2%}"]
        for n, sp in sorted(self.scaled_speedups.items()):
            lines.append(f"  {n} threads: scaled speedup = {sp:.2f}x")
        for r in self.recommendations:
            lines.append(f"  * {r}")
        return "\n".join(lines)


def _run_timed(executable: str, env: Optional[Dict[str, str]] = None,
               timeout: float = 300.0) -> float:
    """Run an executable and return wall-clock time in seconds."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    try:
        start = time.perf_counter()
        subprocess.run(
            executable, shell=True, env=full_env,
            timeout=timeout, capture_output=True,
        )
        return time.perf_counter() - start
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return float('inf')


def profile_scalability(executable: str,
                        thread_counts: Optional[List[int]] = None) -> ScalabilityResult:
    """Profile how an executable scales with thread count.

    The executable should respect OMP_NUM_THREADS or a -t flag for thread count.
    """
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8]

    result = ScalabilityResult()
    base_time: Optional[float] = None

    for n in sorted(thread_counts):
        env = {"OMP_NUM_THREADS": str(n)}
        wall = _run_timed(executable, env=env)

        timing = TimingResult(threads=n, wall_time_seconds=wall)
        if wall > 0 and wall != float('inf'):
            timing.throughput = 1.0 / wall

        result.timings.append(timing)

        if base_time is None and wall != float('inf'):
            base_time = wall

        if base_time and wall != float('inf') and wall > 0:
            speedup = base_time / wall
            result.speedups[n] = speedup
            result.efficiencies[n] = speedup / n

    # Rate scalability
    if result.efficiencies:
        max_threads = max(result.efficiencies.keys())
        max_eff = result.efficiencies.get(max_threads, 0)
        if max_eff > 0.8:
            result.rating = ScalabilityRating.EXCELLENT
        elif max_eff > 0.5:
            result.rating = ScalabilityRating.GOOD
        elif max_eff > 0.3:
            result.rating = ScalabilityRating.FAIR
        else:
            result.rating = ScalabilityRating.POOR

        # Identify bottleneck
        if max_eff < 0.5:
            if len(result.speedups) >= 2:
                vals = list(result.speedups.values())
                if vals[-1] < vals[-2]:
                    result.bottleneck = "Negative scaling at high thread counts — likely contention."
                else:
                    result.bottleneck = "Sub-linear scaling — serial fraction or synchronization overhead."

    return result


def detect_false_sharing(executable: str) -> List[FalseSharing]:
    """Detect false sharing by analyzing struct/array layouts in source or binary.

    Uses perf c2c where available, falls back to static heuristics on source.
    """
    results: List[FalseSharing] = []

    # Try perf c2c if available (Linux only)
    try:
        proc = subprocess.run(
            f"perf c2c record -o /tmp/perf_c2c.data -- {executable}",
            shell=True, capture_output=True, timeout=60,
        )
        if proc.returncode == 0:
            report = subprocess.run(
                "perf c2c report -i /tmp/perf_c2c.data --stdio",
                shell=True, capture_output=True, timeout=30, text=True,
            )
            if report.returncode == 0:
                output = report.stdout
                # Parse hitm lines
                for line in output.split('\n'):
                    m = re.search(r'(\d+)\s+(\d+)\s+.*\s+(\w+)', line)
                    if m and int(m.group(1)) > 100:
                        results.append(FalseSharing(
                            variable_a=m.group(3),
                            variable_b="adjacent",
                            estimated_penalty_pct=min(float(m.group(1)) / 10, 100),
                        ))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # If no perf results, do static analysis on source if it's a source file
    if not results and os.path.isfile(executable) and executable.endswith(('.c', '.cpp', '.h')):
        results = _static_false_sharing_check(executable)

    return results


def _static_false_sharing_check(source_path: str) -> List[FalseSharing]:
    """Heuristic check for false sharing in source code."""
    results: List[FalseSharing] = []
    try:
        with open(source_path) as f:
            source = f.read()
    except OSError:
        return results

    lines = source.split('\n')

    # Pattern: array of small structs accessed by different threads
    struct_pattern = re.compile(r'struct\s+(\w+)\s*\{([^}]*)\}', re.DOTALL)
    for m in struct_pattern.finditer(source):
        name = m.group(1)
        body = m.group(2)
        fields = re.findall(r'(int|long|float|double|char|bool|atomic_\w+)\s+(\w+)', body)
        # Estimate struct size
        type_sizes = {"int": 4, "long": 8, "float": 4, "double": 8, "char": 1, "bool": 1}
        total = sum(type_sizes.get(t.split('_')[0], 8) for t, _ in fields)

        if total > 0 and total < 64:  # smaller than cache line
            # Check if arrays of this struct exist
            if re.search(rf'{name}\s+\w+\s*\[\s*\d+\s*\]', source):
                line_no = source[:m.start()].count('\n') + 1
                results.append(FalseSharing(
                    variable_a=f"struct {name}",
                    variable_b="array element",
                    cache_line_offset=total,
                    estimated_penalty_pct=min((64 - total) / 64 * 100, 80),
                    line=line_no,
                ))

    # Pattern: adjacent global variables accessed in parallel
    global_vars = []
    for i, line in enumerate(lines, 1):
        m = re.match(r'\s*(volatile\s+)?(int|long|float|double|atomic_\w+)\s+(\w+)\s*[;=]', line)
        if m:
            global_vars.append((i, m.group(3)))

    for idx in range(len(global_vars) - 1):
        line_a, var_a = global_vars[idx]
        line_b, var_b = global_vars[idx + 1]
        if line_b - line_a <= 2:  # adjacent declarations
            results.append(FalseSharing(
                variable_a=var_a,
                variable_b=var_b,
                cache_line_offset=0,
                estimated_penalty_pct=30.0,
                line=line_a,
            ))

    return results


def detect_lock_contention(executable: str) -> List[LockContention]:
    """Detect lock contention using profiling tools or static analysis."""
    results: List[LockContention] = []

    # Try mutrace (Linux)
    try:
        proc = subprocess.run(
            f"mutrace -- {executable}",
            shell=True, capture_output=True, timeout=60, text=True,
        )
        if proc.returncode == 0 and proc.stderr:
            output = proc.stderr + proc.stdout
            for line in output.split('\n'):
                m = re.search(
                    r'Mutex\s+#(\d+)\s+.*locked\s+(\d+)\s+.*contended\s+(\d+)', line)
                if m:
                    locked = int(m.group(2))
                    contended = int(m.group(3))
                    ratio = contended / max(locked, 1)
                    level = ContentionLevel.NONE
                    if ratio > 0.5:
                        level = ContentionLevel.SEVERE
                    elif ratio > 0.2:
                        level = ContentionLevel.HIGH
                    elif ratio > 0.05:
                        level = ContentionLevel.MODERATE
                    elif ratio > 0:
                        level = ContentionLevel.LOW

                    results.append(LockContention(
                        lock_name=f"mutex_{m.group(1)}",
                        contention_level=level,
                        wait_time_pct=ratio * 100,
                        acquire_count=locked,
                    ))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: static analysis
    if not results and os.path.isfile(executable):
        results = _static_contention_check(executable)

    return results


def _static_contention_check(source_path: str) -> List[LockContention]:
    """Heuristic contention analysis from source."""
    results: List[LockContention] = []
    try:
        with open(source_path) as f:
            source = f.read()
    except OSError:
        return results

    lines = source.split('\n')
    lock_regions: Dict[str, List[Tuple[int, int]]] = {}

    # Find lock-unlock regions
    lock_pattern = re.compile(r'(pthread_mutex_lock|\.lock\(\))\s*\(?\s*&?\s*(\w+)')
    unlock_pattern = re.compile(r'(pthread_mutex_unlock|\.unlock\(\))\s*\(?\s*&?\s*(\w+)')

    for i, line in enumerate(lines, 1):
        m = lock_pattern.search(line)
        if m:
            name = m.group(2)
            lock_regions.setdefault(name, []).append((i, 0))

        m = unlock_pattern.search(line)
        if m:
            name = m.group(2)
            if name in lock_regions and lock_regions[name]:
                last = lock_regions[name][-1]
                if last[1] == 0:
                    lock_regions[name][-1] = (last[0], i)

    for name, regions in lock_regions.items():
        total_lines = sum(end - start for start, end in regions if end > 0)
        count = len(regions)
        avg_size = total_lines / max(count, 1)

        level = ContentionLevel.NONE
        if avg_size > 50:
            level = ContentionLevel.HIGH
        elif avg_size > 20:
            level = ContentionLevel.MODERATE
        elif avg_size > 5:
            level = ContentionLevel.LOW

        if count > 0:
            results.append(LockContention(
                lock_name=name,
                contention_level=level,
                acquire_count=count,
                hold_time_avg_us=avg_size * 10.0,  # rough estimate
                line=regions[0][0] if regions else 0,
            ))

    return results


def cache_behavior_analysis(executable: str) -> CacheReport:
    """Analyze cache behavior using perf or static heuristics."""
    report = CacheReport()

    # Try perf stat
    try:
        proc = subprocess.run(
            f"perf stat -e cache-references,cache-misses,"
            f"L1-dcache-loads,L1-dcache-load-misses,"
            f"LLC-loads,LLC-load-misses -- {executable}",
            shell=True, capture_output=True, timeout=60, text=True,
        )
        if proc.returncode == 0:
            output = proc.stderr
            refs = _parse_perf_counter(output, "cache-references")
            misses = _parse_perf_counter(output, "cache-misses")
            l1_loads = _parse_perf_counter(output, "L1-dcache-loads")
            l1_misses = _parse_perf_counter(output, "L1-dcache-load-misses")
            llc_loads = _parse_perf_counter(output, "LLC-loads")
            llc_misses = _parse_perf_counter(output, "LLC-load-misses")

            if l1_loads > 0:
                report.l1_hit_rate = 1.0 - (l1_misses / l1_loads)
            if llc_loads > 0:
                report.l3_hit_rate = 1.0 - (llc_misses / llc_loads)
            if refs > 0:
                report.l2_hit_rate = 1.0 - (misses / refs)

            if report.l1_hit_rate < 0.95:
                report.recommendations.append(
                    "L1 cache hit rate below 95% — consider improving data locality."
                )
            if report.l3_hit_rate < 0.9:
                report.recommendations.append(
                    "LLC hit rate below 90% — working set may exceed cache capacity."
                )
            return report
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Fallback: return default with recommendations
    report.recommendations.append(
        "Install perf tools for hardware-level cache analysis."
    )
    report.recommendations.append(
        "Ensure data accessed by the same thread is co-located in memory."
    )
    report.recommendations.append(
        "Pad shared data structures to cache line boundaries (64 bytes) to avoid false sharing."
    )
    return report


def _parse_perf_counter(output: str, counter_name: str) -> int:
    """Parse a perf stat counter value."""
    pattern = re.compile(rf'([\d,]+)\s+{re.escape(counter_name)}')
    m = pattern.search(output)
    if m:
        return int(m.group(1).replace(',', ''))
    return 0


def amdahl_analysis(executable: str,
                    thread_counts: Optional[List[int]] = None) -> AmdahlResult:
    """Estimate the serial fraction and max speedup using Amdahl's law.

    Runs the executable at different thread counts and fits the serial fraction.
    Amdahl: speedup(N) = 1 / (s + (1-s)/N)  where s = serial fraction.
    """
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8]

    # Collect timings
    timings: Dict[int, float] = {}
    for n in thread_counts:
        env = {"OMP_NUM_THREADS": str(n)}
        t = _run_timed(executable, env=env)
        if t != float('inf'):
            timings[n] = t

    if 1 not in timings or len(timings) < 2:
        return AmdahlResult(
            serial_fraction=1.0, max_speedup=1.0,
            recommendations=["Could not collect sufficient timing data."],
        )

    t1 = timings[1]
    speedups = {n: t1 / t for n, t in timings.items() if t > 0}

    # Estimate serial fraction: s = (1/speedup - 1/N) / (1 - 1/N)
    estimates = []
    for n, sp in speedups.items():
        if n > 1:
            s = (1.0 / sp - 1.0 / n) / (1.0 - 1.0 / n)
            estimates.append(max(0.0, min(1.0, s)))

    serial_fraction = sum(estimates) / len(estimates) if estimates else 1.0

    # Compute theoretical speedups
    theoretical: Dict[int, float] = {}
    for n in [1, 2, 4, 8, 16, 32, 64]:
        theoretical[n] = 1.0 / (serial_fraction + (1.0 - serial_fraction) / n)

    max_speedup = 1.0 / serial_fraction if serial_fraction > 0 else float('inf')

    recommendations: List[str] = []
    if serial_fraction > 0.2:
        recommendations.append(
            f"Serial fraction is {serial_fraction:.1%} — focus on parallelizing the serial portion."
        )
    if serial_fraction > 0.5:
        recommendations.append(
            "High serial fraction limits scalability severely — consider algorithmic changes."
        )
    if max_speedup < max(thread_counts):
        recommendations.append(
            f"Max theoretical speedup ({max_speedup:.1f}x) is below the max thread count "
            f"({max(thread_counts)}) — diminishing returns expected."
        )

    return AmdahlResult(
        serial_fraction=serial_fraction,
        max_speedup=min(max_speedup, 1000.0),
        speedup_at=theoretical,
        recommendations=recommendations,
    )


def gustafson_analysis(executable: str,
                       problem_sizes: Optional[List[int]] = None) -> GustafsonResult:
    """Estimate scaled speedup using Gustafson's law.

    Gustafson: scaled_speedup(N) = N - s*(N-1)  where s = serial fraction.
    Unlike Amdahl, assumes problem size scales with processor count.
    """
    if problem_sizes is None:
        problem_sizes = [100, 1000, 10000]

    thread_counts = [1, 2, 4, 8]
    all_timings: Dict[int, Dict[int, float]] = {}

    for size in problem_sizes:
        size_timings: Dict[int, float] = {}
        for n in thread_counts:
            env = {
                "OMP_NUM_THREADS": str(n),
                "PROBLEM_SIZE": str(size),
            }
            t = _run_timed(executable, env=env)
            if t != float('inf'):
                size_timings[n] = t
        if size_timings:
            all_timings[size] = size_timings

    # Estimate serial fraction from parallel runs
    serial_fractions: List[float] = []
    for size, timings in all_timings.items():
        if 1 in timings:
            t1 = timings[1]
            for n, tn in timings.items():
                if n > 1 and tn > 0:
                    sp = t1 / tn
                    s = (n - sp) / (n - 1)
                    serial_fractions.append(max(0.0, min(1.0, s)))

    serial_fraction = (sum(serial_fractions) / len(serial_fractions)
                       if serial_fractions else 0.5)

    # Compute Gustafson scaled speedups
    scaled: Dict[int, float] = {}
    for n in [1, 2, 4, 8, 16, 32, 64]:
        scaled[n] = n - serial_fraction * (n - 1)

    # Problem size scaling
    problem_scaling: Dict[int, float] = {}
    for size, timings in all_timings.items():
        if len(timings) >= 2:
            max_n = max(timings.keys())
            if 1 in timings and max_n > 1:
                problem_scaling[size] = timings[1] / timings[max_n]

    recommendations: List[str] = []
    if serial_fraction < 0.1:
        recommendations.append(
            "Low serial fraction — problem scales well with more processors."
        )
    elif serial_fraction > 0.3:
        recommendations.append(
            f"Serial fraction of {serial_fraction:.1%} limits scaled speedup."
        )

    if problem_scaling:
        sizes = sorted(problem_scaling.keys())
        if len(sizes) >= 2:
            if problem_scaling[sizes[-1]] > problem_scaling[sizes[0]]:
                recommendations.append(
                    "Larger problem sizes show better scaling — Gustafson's law applies well."
                )

    return GustafsonResult(
        serial_fraction=serial_fraction,
        scaled_speedups=scaled,
        problem_size_scaling=problem_scaling,
        recommendations=recommendations,
    )
