"""
Report generation for CoaCert evaluation results.

Produces formatted output in console text, JSON, LaTeX, and Markdown.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TextIO

from .metrics import (
    AggregatedMetrics,
    PipelineMetrics,
    aggregate_metrics,
)
from .timing import TimingStats, format_duration


@dataclass
class SystemInfo:
    """Information about the system used for benchmarking."""
    os_name: str = ""
    os_version: str = ""
    cpu: str = ""
    cpu_count: int = 0
    memory_gb: float = 0.0
    python_version: str = ""
    timestamp: str = ""

    @classmethod
    def collect(cls) -> "SystemInfo":
        uname = platform.uname()
        try:
            import multiprocessing
            cpus = multiprocessing.cpu_count()
        except Exception:
            cpus = os.cpu_count() or 0
        mem_gb = 0.0
        try:
            import resource
            # rough fallback
            mem_gb = 0.0
        except Exception:
            pass
        # Try reading total memory
        try:
            if sys.platform == "linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            mem_gb = kb / (1024 * 1024)
                            break
            elif sys.platform == "darwin":
                import subprocess
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True
                ).strip()
                mem_gb = int(out) / (1024**3)
        except Exception:
            pass

        return cls(
            os_name=uname.system,
            os_version=uname.release,
            cpu=uname.machine,
            cpu_count=cpus,
            memory_gb=round(mem_gb, 1),
            python_version=platform.python_version(),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "os": f"{self.os_name} {self.os_version}",
            "cpu": self.cpu,
            "cpu_count": self.cpu_count,
            "memory_gb": self.memory_gb,
            "python_version": self.python_version,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkSummary:
    """Per-benchmark summary for the report."""
    name: str
    runs: int
    original_states: int
    quotient_states: int
    state_ratio: float
    original_transitions: int
    quotient_transitions: int
    transition_ratio: float
    mean_total_time: float
    mean_queries: float
    witness_size_bytes: int
    correctness_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "runs": self.runs,
            "original_states": self.original_states,
            "quotient_states": self.quotient_states,
            "state_ratio": self.state_ratio,
            "original_transitions": self.original_transitions,
            "quotient_transitions": self.quotient_transitions,
            "transition_ratio": self.transition_ratio,
            "mean_total_time": self.mean_total_time,
            "mean_queries": self.mean_queries,
            "witness_size_bytes": self.witness_size_bytes,
            "correctness_score": self.correctness_score,
        }


def _summarize_benchmark(
    name: str, runs: Sequence[PipelineMetrics]
) -> BenchmarkSummary:
    agg = aggregate_metrics(runs)
    last = runs[-1] if runs else PipelineMetrics()
    return BenchmarkSummary(
        name=name,
        runs=len(runs),
        original_states=last.state_space.original_states,
        quotient_states=last.state_space.quotient_states,
        state_ratio=agg.state_compression_ratio,
        original_transitions=last.state_space.original_transitions,
        quotient_transitions=last.state_space.quotient_transitions,
        transition_ratio=agg.transition_compression_ratio,
        mean_total_time=agg.mean_total_time,
        mean_queries=agg.mean_queries,
        witness_size_bytes=last.witness.witness_size_bytes,
    )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _pad(s: str, width: int) -> str:
    return s[:width].ljust(width)


class ReportGenerator:
    """Generate evaluation reports in multiple output formats.

    Usage::

        rg = ReportGenerator()
        rg.add_benchmark("mutex_2", runs_mutex2)
        rg.add_benchmark("peterson_3", runs_peterson3)
        print(rg.console_report())
        rg.write_json("results.json")
    """

    def __init__(self, title: str = "CoaCert Evaluation Report") -> None:
        self._title = title
        self._benchmarks: Dict[str, List[PipelineMetrics]] = {}
        self._system_info = SystemInfo.collect()
        self._notes: List[str] = []

    def add_benchmark(
        self, name: str, runs: Sequence[PipelineMetrics]
    ) -> None:
        self._benchmarks[name] = list(runs)

    def add_note(self, note: str) -> None:
        self._notes.append(note)

    @property
    def benchmark_names(self) -> List[str]:
        return list(self._benchmarks.keys())

    def _summaries(self) -> List[BenchmarkSummary]:
        return [
            _summarize_benchmark(name, runs)
            for name, runs in self._benchmarks.items()
        ]

    def _best_compression(self) -> Optional[BenchmarkSummary]:
        sums = self._summaries()
        if not sums:
            return None
        return min(sums, key=lambda s: s.state_ratio)

    def _fastest_verification(self) -> Optional[BenchmarkSummary]:
        sums = self._summaries()
        if not sums:
            return None
        return min(sums, key=lambda s: s.mean_total_time)

    # -- Console report ------------------------------------------------------

    def console_report(self) -> str:
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append(f"  {self._title}")
        lines.append("=" * 80)

        # System info
        si = self._system_info
        lines.append(f"\nSystem: {si.os_name} {si.os_version}, "
                      f"{si.cpu} ({si.cpu_count} CPUs), "
                      f"{si.memory_gb} GB RAM, Python {si.python_version}")
        lines.append(f"Date:   {si.timestamp}\n")

        # Overview table
        sums = self._summaries()
        if not sums:
            lines.append("No benchmark results.")
            return "\n".join(lines)

        hdr = (
            f"{'Benchmark':<24} "
            f"{'|S|':>8} {'|S/~|':>8} {'Ratio':>8} "
            f"{'|T|':>8} {'|T/~|':>8} {'Ratio':>8} "
            f"{'Time':>10} {'Queries':>8}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for s in sums:
            lines.append(
                f"{s.name:<24} "
                f"{s.original_states:>8} {s.quotient_states:>8} "
                f"{s.state_ratio:>8.4f} "
                f"{s.original_transitions:>8} {s.quotient_transitions:>8} "
                f"{s.transition_ratio:>8.4f} "
                f"{format_duration(s.mean_total_time):>10} "
                f"{s.mean_queries:>8.0f}"
            )
        lines.append("-" * len(hdr))

        # Highlights
        best = self._best_compression()
        fastest = self._fastest_verification()
        if best:
            lines.append(
                f"\nBest compression:     {best.name} "
                f"(ratio={best.state_ratio:.4f})"
            )
        if fastest:
            lines.append(
                f"Fastest verification: {fastest.name} "
                f"({format_duration(fastest.mean_total_time)})"
            )

        # Notes
        if self._notes:
            lines.append("\nNotes:")
            for n in self._notes:
                lines.append(f"  - {n}")

        lines.append("")
        return "\n".join(lines)

    # -- JSON report ---------------------------------------------------------

    def json_report(self, indent: int = 2) -> str:
        data: Dict[str, Any] = {
            "title": self._title,
            "system": self._system_info.to_dict(),
            "benchmarks": {},
            "notes": self._notes,
        }
        for name, runs in self._benchmarks.items():
            summary = _summarize_benchmark(name, runs)
            agg = aggregate_metrics(runs)
            data["benchmarks"][name] = {
                "summary": summary.to_dict(),
                "aggregated": agg.to_dict(),
                "runs": [r.to_dict() for r in runs],
            }
        best = self._best_compression()
        fastest = self._fastest_verification()
        data["highlights"] = {
            "best_compression": best.name if best else None,
            "fastest_verification": fastest.name if fastest else None,
        }
        return json.dumps(data, indent=indent)

    def write_json(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            f.write(self.json_report(indent))

    # -- LaTeX report --------------------------------------------------------

    def latex_table(self) -> str:
        """Generate a LaTeX table suitable for paper inclusion."""
        sums = self._summaries()
        lines: List[str] = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{" + self._title + "}")
        lines.append(r"\label{tab:evaluation}")
        lines.append(
            r"\begin{tabular}{l r r r r r r r}"
        )
        lines.append(r"\toprule")
        lines.append(
            r"Benchmark & $|S|$ & $|S/{\sim}|$ & Ratio "
            r"& $|T|$ & $|T/{\sim}|$ & Ratio & Time \\"
        )
        lines.append(r"\midrule")

        for s in sums:
            time_str = format_duration(s.mean_total_time)
            lines.append(
                f"  {_latex_escape(s.name)} & "
                f"{s.original_states} & {s.quotient_states} & "
                f"{s.state_ratio:.4f} & "
                f"{s.original_transitions} & {s.quotient_transitions} & "
                f"{s.transition_ratio:.4f} & "
                f"{time_str} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    # -- Markdown report -----------------------------------------------------

    def markdown_report(self) -> str:
        sums = self._summaries()
        lines: List[str] = []
        lines.append(f"# {self._title}\n")

        si = self._system_info
        lines.append(
            f"**System:** {si.os_name} {si.os_version}, "
            f"{si.cpu} ({si.cpu_count} CPUs), "
            f"{si.memory_gb} GB RAM, Python {si.python_version}  "
        )
        lines.append(f"**Date:** {si.timestamp}\n")

        if not sums:
            lines.append("_No benchmark results._")
            return "\n".join(lines)

        # Table
        lines.append(
            "| Benchmark | |S| | |S/~| | State Ratio | "
            "|T| | |T/~| | Trans Ratio | Time | Queries |"
        )
        lines.append(
            "|-----------|-----|-------|-------------|"
            "-----|-------|-------------|------|---------|"
        )
        for s in sums:
            lines.append(
                f"| {s.name} | {s.original_states} | {s.quotient_states} | "
                f"{s.state_ratio:.4f} | "
                f"{s.original_transitions} | {s.quotient_transitions} | "
                f"{s.transition_ratio:.4f} | "
                f"{format_duration(s.mean_total_time)} | "
                f"{s.mean_queries:.0f} |"
            )

        # Highlights
        best = self._best_compression()
        fastest = self._fastest_verification()
        lines.append("\n## Highlights\n")
        if best:
            lines.append(
                f"- **Best compression:** {best.name} "
                f"(ratio = {best.state_ratio:.4f})"
            )
        if fastest:
            lines.append(
                f"- **Fastest verification:** {fastest.name} "
                f"({format_duration(fastest.mean_total_time)})"
            )

        if self._notes:
            lines.append("\n## Notes\n")
            for n in self._notes:
                lines.append(f"- {n}")

        lines.append("")
        return "\n".join(lines)

    def write_markdown(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.markdown_report())

    def write_latex(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.latex_table())

    # -- comparison table (two runs) -----------------------------------------

    def comparison_table(
        self,
        other_benchmarks: Dict[str, List[PipelineMetrics]],
        label_self: str = "Baseline",
        label_other: str = "New",
    ) -> str:
        """Console table comparing self vs other benchmark runs."""
        lines: List[str] = []
        lines.append(f"Comparison: {label_self} vs {label_other}")
        hdr = (
            f"{'Benchmark':<24} "
            f"{'Ratio(' + label_self + ')':>16} "
            f"{'Ratio(' + label_other + ')':>16} "
            f"{'Time(' + label_self + ')':>14} "
            f"{'Time(' + label_other + ')':>14} "
            f"{'Speedup':>10}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))

        all_names = sorted(set(self._benchmarks.keys()) | set(other_benchmarks.keys()))
        for name in all_names:
            s_runs = self._benchmarks.get(name, [])
            o_runs = other_benchmarks.get(name, [])
            s_agg = aggregate_metrics(s_runs) if s_runs else None
            o_agg = aggregate_metrics(o_runs) if o_runs else None
            s_ratio = f"{s_agg.state_compression_ratio:.4f}" if s_agg else "N/A"
            o_ratio = f"{o_agg.state_compression_ratio:.4f}" if o_agg else "N/A"
            s_time = format_duration(s_agg.mean_total_time) if s_agg else "N/A"
            o_time = format_duration(o_agg.mean_total_time) if o_agg else "N/A"
            if s_agg and o_agg and o_agg.mean_total_time > 0:
                speedup = f"{s_agg.mean_total_time / o_agg.mean_total_time:.2f}x"
            else:
                speedup = "N/A"
            lines.append(
                f"{name:<24} "
                f"{s_ratio:>16} "
                f"{o_ratio:>16} "
                f"{s_time:>14} "
                f"{o_time:>14} "
                f"{speedup:>10}"
            )
        lines.append("-" * len(hdr))
        return "\n".join(lines)


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    for ch in ("_", "&", "%", "#", "{", "}"):
        s = s.replace(ch, "\\" + ch)
    return s
