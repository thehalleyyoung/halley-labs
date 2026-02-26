"""Performance profiling for BioProver CEGAR verification.

Provides phase-level timing, memory tracking, scalability analysis,
regression detection, and formatted reporting for verification runs.
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from bioprover.cegar import CEGARConfig, CEGAREngine, CEGARStatistics, VerificationResult, VerificationStatus
from bioprover.models import BioModel
from bioprover.temporal import STLFormula

try:
    import resource as _resource
    def _get_memory_mb() -> Tuple[float, float]:
        u = _resource.getrusage(_resource.RUSAGE_SELF)
        import sys
        rss = u.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else u.ru_maxrss / 1024
        return rss, 0.0
except ImportError:
    _resource = None
    def _get_memory_mb() -> Tuple[float, float]:
        return 0.0, 0.0

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

def _format_duration(s: float) -> str:
    if s < 1e-3: return f"{s*1e6:.1f}µs"
    if s < 1.0:  return f"{s*1e3:.1f}ms"
    if s < 60:   return f"{s:.2f}s"
    return f"{int(s//60)}m {s%60:.1f}s"

def _format_memory(mb: float) -> str:
    if mb < 1.0:    return f"{mb*1024:.0f} KB"
    if mb < 1024.0: return f"{mb:.1f} MB"
    return f"{mb/1024:.2f} GB"

# ---------------------------------------------------------------------------
class PhaseTimer:
    """Context manager for timing a named code phase."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.wall_time: float = 0.0
        self._running = False

    def __enter__(self) -> PhaseTimer:
        self.start_time = time.monotonic()
        self._running = True
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end_time = time.monotonic()
        self.wall_time = self.end_time - self.start_time
        self._running = False

    @property
    def elapsed(self) -> float:
        return (time.monotonic() - self.start_time) if self._running else self.wall_time

# ---------------------------------------------------------------------------
@dataclass
class MemorySnapshot:
    timestamp: float
    rss_mb: float
    vms_mb: float
    phase: str

@dataclass
class PhaseMetrics:
    name: str
    wall_time: float
    memory_peak_mb: float
    call_count: int
    sub_phases: List[PhaseMetrics] = field(default_factory=list)

# ---------------------------------------------------------------------------
class ProfilingSession:
    """Collects timing, memory, and solver metrics for a verification run."""

    def __init__(self, name: str, track_memory: bool = True, memory_interval: float = 0.1) -> None:
        self.name, self.track_memory, self.memory_interval = name, track_memory, memory_interval
        self._start = time.monotonic()
        self._phase_timers: List[PhaseTimer] = []
        self._phase_counts: Dict[str, int] = {}
        self._memory_snapshots: List[MemorySnapshot] = []
        self._cegar_iterations: List[Dict[str, Any]] = []
        self._solver_queries: List[Dict[str, Any]] = []
        self._current_phase: Optional[str] = None

    def phase(self, name: str) -> PhaseTimer:
        """Return a context manager that records wall time for *name*."""
        timer = PhaseTimer(name)
        self._phase_timers.append(timer)
        self._phase_counts[name] = self._phase_counts.get(name, 0) + 1
        session = self
        orig_enter, orig_exit = timer.__enter__, timer.__exit__
        def _enter(s=session, t=timer):
            s._current_phase = name
            if s.track_memory: s.record_memory()
            return orig_enter()
        def _exit(*exc, s=session, t=timer):
            if s.track_memory: s.record_memory()
            s._current_phase = None
            return orig_exit(*exc)
        timer.__enter__ = _enter   # type: ignore[assignment]
        timer.__exit__  = _exit    # type: ignore[assignment]
        return timer

    def record_memory(self) -> MemorySnapshot:
        rss, vms = _get_memory_mb()
        snap = MemorySnapshot(time.monotonic() - self._start, rss, vms, self._current_phase or "unknown")
        self._memory_snapshots.append(snap)
        return snap

    def record_cegar_iteration(self, iteration: int, stats_snapshot: Dict[str, Any]) -> None:
        self._cegar_iterations.append({"iteration": iteration, "timestamp": time.monotonic() - self._start, **stats_snapshot})

    def record_solver_query(self, query_type: str, duration: float, result: str) -> None:
        self._solver_queries.append({"query_type": query_type, "duration": duration, "result": result, "timestamp": time.monotonic() - self._start})

    def get_phase_breakdown(self) -> Dict[str, PhaseMetrics]:
        times: Dict[str, float] = {}
        for t in self._phase_timers:
            times[t.name] = times.get(t.name, 0.0) + t.wall_time
        mem: Dict[str, float] = {}
        for s in self._memory_snapshots:
            mem[s.phase] = max(mem.get(s.phase, 0.0), s.rss_mb)
        return {n: PhaseMetrics(n, wt, mem.get(n, 0.0), self._phase_counts.get(n, 1)) for n, wt in times.items()}

    def total_time(self) -> float:
        return time.monotonic() - self._start

    def peak_memory(self) -> float:
        return max((s.rss_mb for s in self._memory_snapshots), default=0.0)

    def summary(self) -> str:
        lines = [f"=== Profiling: {self.name} ===",
                 f"Total time : {_format_duration(self.total_time())}",
                 f"Peak memory: {_format_memory(self.peak_memory())}"]
        bd = self.get_phase_breakdown()
        if bd:
            lines.append("\nPhase breakdown:")
            tot = max(self.total_time(), 1e-9)
            for pm in sorted(bd.values(), key=lambda p: p.wall_time, reverse=True):
                lines.append(f"  {pm.name:<25s} {_format_duration(pm.wall_time):>10s}  ({pm.wall_time/tot*100:5.1f}%)  calls={pm.call_count}")
        if self._solver_queries:
            total_sq = sum(q["duration"] for q in self._solver_queries)
            lines += [f"\nSolver queries: {len(self._solver_queries)}", f"  Total solver time: {_format_duration(total_sq)}"]
        return "\n".join(lines)

# ---------------------------------------------------------------------------
class ScalabilityAnalysis:
    """Measure how verification time scales with model / spec dimensions."""

    def __init__(self) -> None:
        self._results: Dict[str, List[Tuple[int, float]]] = {}

    @staticmethod
    def _run_benchmark(model: BioModel, spec: STLFormula, config: CEGARConfig, runs: int) -> float:
        times: List[float] = []
        for _ in range(runs):
            engine = CEGAREngine(model, spec, config)
            t0 = time.monotonic()
            engine.verify()
            times.append(time.monotonic() - t0)
        return sum(times) / len(times)

    def time_vs_species(self, models: List[BioModel], spec_factory: Callable[[BioModel], STLFormula],
                        config: CEGARConfig, runs_per_point: int = 3) -> List[Tuple[int, float]]:
        pts = [(len(m.species), self._run_benchmark(m, spec_factory(m), config, runs_per_point)) for m in models]
        self._results["species"] = pts
        return pts

    def time_vs_parameters(self, models: List[BioModel], spec_factory: Callable[[BioModel], STLFormula],
                           config: CEGARConfig, runs_per_point: int = 3) -> List[Tuple[int, float]]:
        pts = [(len(m.parameters), self._run_benchmark(m, spec_factory(m), config, runs_per_point)) for m in models]
        self._results["parameters"] = pts
        return pts

    def time_vs_horizon(self, model: BioModel, spec_factory: Callable[[float], STLFormula],
                        horizons: List[float], config: CEGARConfig, runs_per_point: int = 3) -> List[Tuple[int, float]]:
        pts = [(int(h), self._run_benchmark(model, spec_factory(h), config, runs_per_point)) for h in horizons]
        self._results["horizon"] = pts
        return pts

    def time_vs_spec_complexity(self, model: BioModel, specs: List[STLFormula],
                                config: CEGARConfig, runs_per_point: int = 3) -> List[Tuple[int, float]]:
        pts = [(getattr(s, "depth", 1) + getattr(s, "size", 1),
                self._run_benchmark(model, s, config, runs_per_point)) for s in specs]
        self._results["spec_complexity"] = pts
        return pts

    @staticmethod
    def fit_complexity(data_points: List[Tuple[int, float]]) -> Dict[str, Any]:
        """Fit polynomial / exponential models; return best-fit info with R² values."""
        if not _HAS_NUMPY or len(data_points) < 3:
            return {"error": "numpy required and >= 3 data points needed"}
        xs = np.array([p[0] for p in data_points], dtype=float)
        ys = np.array([p[1] for p in data_points], dtype=float)
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        if ss_tot == 0:
            return {"error": "no variance in data"}
        results: Dict[str, Any] = {}
        for deg in (1, 2):
            c = np.polyfit(xs, ys, deg)
            ss_res = float(np.sum((ys - np.polyval(c, xs)) ** 2))
            results[f"poly_{deg}"] = {"coefficients": c.tolist(), "r_squared": 1 - ss_res / ss_tot}
        if np.all(ys > 0):
            c = np.polyfit(xs, np.log(ys), 1)
            ss_res = float(np.sum((ys - np.exp(np.polyval(c, xs))) ** 2))
            results["exponential"] = {"a": float(np.exp(c[1])), "b": float(c[0]), "r_squared": 1 - ss_res / ss_tot}
        results["best_fit"] = max((k for k in results if k != "best_fit"), key=lambda k: results[k].get("r_squared", -1))
        return results

    @staticmethod
    def ascii_plot(data_points: List[Tuple[int, float]], x_label: str = "x",
                   y_label: str = "y", title: str = "", width: int = 60, height: int = 20) -> str:
        if not data_points: return "(no data)"
        xs, ys = [p[0] for p in data_points], [p[1] for p in data_points]
        xn, xx, yn, yx = min(xs), max(xs), min(ys), max(ys)
        xr, yr = max(xx - xn, 1), max(yx - yn, 1e-9)
        grid = [[" "] * width for _ in range(height)]
        for x, y in zip(xs, ys):
            c = min(width - 1, max(0, int((x - xn) / xr * (width - 1))))
            r = min(height - 1, max(0, height - 1 - int((y - yn) / yr * (height - 1))))
            grid[r][c] = "*"
        lines: List[str] = []
        if title: lines.append(title.center(width + 8))
        for i, row in enumerate(grid):
            lbl = f"{yx:8.2f}" if i == 0 else (f"{yn:8.2f}" if i == height - 1 else "        ")
            lines.append(f"{lbl} |{''.join(row)}|")
        lines += ["         " + "-" * width, f"         {x_label}: {xn}..{xx}    {y_label}"]
        return "\n".join(lines)

# ---------------------------------------------------------------------------
class RegressionDetector:
    """Compare current timings against a saved baseline to detect regressions."""

    def __init__(self, baseline_file: Optional[str] = None, threshold_pct: float = 20.0) -> None:
        self.threshold_pct = threshold_pct
        self._baseline: Dict[str, float] = {}
        if baseline_file is not None:
            self.load_baseline(baseline_file)

    def load_baseline(self, path: str) -> None:
        with open(path) as f:
            self._baseline = json.load(f)

    def save_baseline(self, results: Dict[str, float], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    def check(self, current: Dict[str, float]) -> List[str]:
        """Return warnings for benchmarks that regressed beyond threshold."""
        return [f"REGRESSION {n}: {_format_duration(c)} vs baseline {_format_duration(b)} (+{(c/b-1)*100:.1f}%)"
                for n, c in current.items()
                if (b := self._baseline.get(n)) and b > 0 and c / b > 1 + self.threshold_pct / 100]

    def detect_regressions(self, current_results: Dict[str, float],
                           baseline_results: Dict[str, float]) -> List[Dict[str, Any]]:
        regs: List[Dict[str, Any]] = []
        for name, cur in current_results.items():
            base = baseline_results.get(name)
            if not base or base <= 0: continue
            ratio = cur / base
            if ratio > 1 + self.threshold_pct / 100:
                regs.append({"benchmark": name, "baseline": base, "current": cur,
                             "ratio": ratio, "regression_pct": (ratio - 1) * 100})
        return regs

# ---------------------------------------------------------------------------
class ProfilingReport:
    """Generate text and LaTeX reports from a profiling session."""

    def __init__(self, session: ProfilingSession, scalability: Optional[ScalabilityAnalysis] = None) -> None:
        self.session, self.scalability = session, scalability

    def text_report(self) -> str:
        parts = [f"BioProver Profiling Report: {self.session.name}", "=" * 60]
        bd = self.session.get_phase_breakdown()
        if bd:
            tot = max(self.session.total_time(), 1e-9)
            h = ["Phase", "Wall Time", "% Total", "Peak Mem", "Calls"]
            rows = [[pm.name, _format_duration(pm.wall_time), f"{pm.wall_time/tot*100:.1f}%",
                     _format_memory(pm.memory_peak_mb), str(pm.call_count)]
                    for pm in sorted(bd.values(), key=lambda p: p.wall_time, reverse=True)]
            parts += ["\nPhase Breakdown", self._format_table(h, rows, "lrrrl")]
        parts.append(f"\nPeak Memory: {_format_memory(self.session.peak_memory())}")
        if self.session._cegar_iterations:
            parts.append(f"\nCEGAR Iterations: {len(self.session._cegar_iterations)}")
            for it in self.session._cegar_iterations:
                parts.append(f"  iter {it['iteration']:3d}  t={it['timestamp']:.3f}s")
        if self.session._solver_queries:
            by_type: Dict[str, List[float]] = {}
            for q in self.session._solver_queries:
                by_type.setdefault(q["query_type"], []).append(q["duration"])
            parts.append("\nSolver Query Statistics")
            h = ["Type", "Count", "Total", "Mean", "Max"]
            rows = [[qt, str(len(d)), _format_duration(sum(d)),
                     _format_duration(sum(d)/len(d)), _format_duration(max(d))]
                    for qt, d in sorted(by_type.items())]
            parts.append(self._format_table(h, rows, "lrrrr"))
        return "\n".join(parts)

    def latex_report(self) -> str:
        lines = [r"% BioProver profiling — auto-generated", ""]
        bd = self.session.get_phase_breakdown()
        if bd:
            tot = max(self.session.total_time(), 1e-9)
            lines += [r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{lrrr}", r"\toprule",
                      r"Phase & Wall Time (s) & \% Total & Peak Mem (MB) \\", r"\midrule"]
            for pm in sorted(bd.values(), key=lambda p: p.wall_time, reverse=True):
                lines.append(f"{pm.name} & {pm.wall_time:.3f} & {pm.wall_time/tot*100:.1f} & {pm.memory_peak_mb:.1f} \\\\")
            lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Phase timing breakdown}", r"\end{table}"]
        if self.scalability and self.scalability._results:
            for dim, pts in self.scalability._results.items():
                lines += ["", f"% pgfplots data: time vs {dim}",
                          r"\begin{filecontents}{scalability_" + dim + ".dat}", "x y"]
                lines += [f"{x} {y:.6f}" for x, y in pts]
                lines.append(r"\end{filecontents}")
        return "\n".join(lines)

    @staticmethod
    def _format_table(headers: List[str], rows: List[List[str]], alignments: str = "") -> str:
        all_rows = [headers] + rows
        widths = [max(len(r[c]) for r in all_rows) for c in range(len(headers))]
        sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        def fmt(row: List[str]) -> str:
            cells = [(v.rjust(w) if (i < len(alignments) and alignments[i] == "r") else v.ljust(w))
                     for i, (v, w) in enumerate(zip(row, widths))]
            return "| " + " | ".join(cells) + " |"
        return "\n".join([sep, fmt(headers), sep] + [fmt(r) for r in rows] + [sep])

    @staticmethod
    def _format_duration(seconds: float) -> str:
        return _format_duration(seconds)

    @staticmethod
    def _format_memory(mb: float) -> str:
        return _format_memory(mb)
