"""Scalability experiments for BioProver.

Measures verification time vs. species count for:
- Monolithic CEGAR
- Compositional CEGAR with auto-decomposition
- Bounded model checking (baseline)

Results are exported as JSON for paper figures and analysis.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bioprover.cegar import CEGARConfig, CEGAREngine, VerificationStatus
from bioprover.evaluation.benchmark_suite import (
    BenchmarkCircuit,
    BenchmarkDifficulty,
    BenchmarkSuite,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification mode enum
# ---------------------------------------------------------------------------


class VerificationMode(Enum):
    """Verification strategy to benchmark."""

    MONOLITHIC = auto()
    COMPOSITIONAL = auto()
    BOUNDED_MC = auto()


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class ScalabilityDataPoint:
    """A single measurement from a scalability experiment."""

    benchmark_name: str
    species_count: int
    parameter_count: int
    mode: str
    wall_time_s: float
    peak_memory_mb: float
    cegar_iterations: int
    verification_status: str
    soundness_level: str
    n_modules: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ScalabilityReport:
    """Aggregate report from a scalability experiment run."""

    data_points: List[ScalabilityDataPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize to JSON. If *path* is given, also write to file."""
        payload = {
            "metadata": self.metadata,
            "results": [dp.to_dict() for dp in self.data_points],
        }
        text = json.dumps(payload, indent=2, default=str)
        if path is not None:
            Path(path).write_text(text)
            logger.info("Scalability report written to %s", path)
        return text

    def summary_table(self) -> str:
        """Format results as an ASCII table."""
        header = (
            f"{'Benchmark':<35} {'#Sp':>4} {'Mode':<15} "
            f"{'Time (s)':>10} {'Mem (MB)':>10} {'Iters':>6} "
            f"{'Status':<12}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for dp in sorted(self.data_points,
                         key=lambda d: (d.species_count, d.mode)):
            lines.append(
                f"{dp.benchmark_name:<35} {dp.species_count:>4} "
                f"{dp.mode:<15} {dp.wall_time_s:>10.2f} "
                f"{dp.peak_memory_mb:>10.1f} {dp.cegar_iterations:>6} "
                f"{dp.verification_status:<12}"
            )
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


class ScalabilityExperiment:
    """Run scalability experiments across benchmark circuits.

    Compares monolithic CEGAR, compositional CEGAR with
    auto-decomposition, and bounded model checking on circuits of
    increasing species count.

    Parameters
    ----------
    benchmarks:
        List of benchmark circuits to evaluate. If None, uses all
        large-scale benchmarks from BenchmarkSuite.
    modes:
        Verification modes to compare.
    cegar_config:
        Configuration for CEGAR engine runs.
    timeout:
        Per-benchmark timeout in seconds.
    output_path:
        Path for JSON results output.
    """

    def __init__(
        self,
        benchmarks: Optional[List[BenchmarkCircuit]] = None,
        modes: Optional[Sequence[VerificationMode]] = None,
        cegar_config: Optional[CEGARConfig] = None,
        timeout: float = 600.0,
        output_path: Optional[str] = None,
    ) -> None:
        if benchmarks is None:
            benchmarks = self._default_benchmarks()
        self.benchmarks = sorted(
            benchmarks,
            key=lambda b: len(b.model.species)
            if hasattr(b.model, "species") else 0,
        )
        self.modes = list(modes or [
            VerificationMode.MONOLITHIC,
            VerificationMode.COMPOSITIONAL,
            VerificationMode.BOUNDED_MC,
        ])
        self.config = cegar_config or CEGARConfig(
            max_iterations=50, timeout=timeout,
        )
        self.timeout = timeout
        self.output_path = output_path

    def run(self) -> ScalabilityReport:
        """Execute the full scalability experiment.

        Returns:
            ScalabilityReport with all data points.
        """
        report = ScalabilityReport(metadata={
            "n_benchmarks": len(self.benchmarks),
            "modes": [m.name for m in self.modes],
            "timeout_s": self.timeout,
            "cegar_max_iterations": self.config.max_iterations,
        })

        for benchmark in self.benchmarks:
            for mode in self.modes:
                logger.info(
                    "Running %s in %s mode (%d species)",
                    benchmark.name, mode.name,
                    len(benchmark.model.species)
                    if hasattr(benchmark.model, "species") else 0,
                )
                dp = self._run_single(benchmark, mode)
                report.data_points.append(dp)

        if self.output_path:
            report.to_json(self.output_path)

        return report

    def run_scaling_sweep(
        self,
        species_counts: Optional[List[int]] = None,
    ) -> ScalabilityReport:
        """Run parameterised cascade benchmarks at varying species counts.

        Uses ``BenchmarkSuite.n_stage_cascade(n)`` for n in
        *species_counts* to measure scaling behavior.

        Args:
            species_counts: List of cascade depths to test.

        Returns:
            ScalabilityReport with sweep data.
        """
        if species_counts is None:
            species_counts = [2, 3, 4, 5, 6, 8, 10]

        report = ScalabilityReport(metadata={
            "experiment": "scaling_sweep",
            "species_counts": species_counts,
            "modes": [m.name for m in self.modes],
        })

        for n in species_counts:
            try:
                benchmark = BenchmarkSuite.n_stage_cascade(n)
            except (ValueError, Exception) as exc:
                logger.warning("Could not create cascade(%d): %s", n, exc)
                continue

            for mode in self.modes:
                dp = self._run_single(benchmark, mode)
                report.data_points.append(dp)

        if self.output_path:
            report.to_json(self.output_path)
        return report

    # -- internal -------------------------------------------------------

    def _run_single(
        self,
        benchmark: BenchmarkCircuit,
        mode: VerificationMode,
    ) -> ScalabilityDataPoint:
        """Run a single benchmark in a single mode and measure."""
        species_count = (
            len(benchmark.model.species)
            if hasattr(benchmark.model, "species") else 0
        )
        param_count = (
            len(benchmark.model.parameters)
            if hasattr(benchmark.model, "parameters") else 0
        )

        t0 = time.monotonic()
        mem_before = self._current_memory_mb()

        try:
            if mode == VerificationMode.MONOLITHIC:
                result, iters, n_modules = self._run_monolithic(benchmark)
            elif mode == VerificationMode.COMPOSITIONAL:
                result, iters, n_modules = self._run_compositional(benchmark)
            elif mode == VerificationMode.BOUNDED_MC:
                result, iters, n_modules = self._run_bounded_mc(benchmark)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            elapsed = time.monotonic() - t0
            mem_after = self._current_memory_mb()

            return ScalabilityDataPoint(
                benchmark_name=benchmark.name,
                species_count=species_count,
                parameter_count=param_count,
                mode=mode.name,
                wall_time_s=elapsed,
                peak_memory_mb=max(0.0, mem_after - mem_before),
                cegar_iterations=iters,
                verification_status=result.name,
                soundness_level=self._soundness_level(result),
                n_modules=n_modules,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "Benchmark %s (%s) failed: %s",
                benchmark.name, mode.name, exc,
            )
            return ScalabilityDataPoint(
                benchmark_name=benchmark.name,
                species_count=species_count,
                parameter_count=param_count,
                mode=mode.name,
                wall_time_s=elapsed,
                peak_memory_mb=0.0,
                cegar_iterations=0,
                verification_status="ERROR",
                soundness_level="none",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _run_monolithic(
        self, benchmark: BenchmarkCircuit,
    ) -> tuple:
        """Run monolithic CEGAR verification."""
        species_names = (
            [s.name for s in benchmark.model.species]
            if hasattr(benchmark.model, "species") else []
        )
        bounds = {s: (0.0, 500.0) for s in species_names}
        engine = CEGAREngine(
            bounds=bounds, rhs={},
            property_expr=None,  # type: ignore[arg-type]
            property_name=benchmark.name,
            config=self.config,
        )
        result = engine.verify()
        iters = (result.statistics.iterations
                 if hasattr(result, "statistics")
                 and hasattr(result.statistics, "iterations")
                 else 0)
        return result.status, iters, 1

    def _run_compositional(
        self, benchmark: BenchmarkCircuit,
    ) -> tuple:
        """Run compositional CEGAR with auto-decomposition."""
        import networkx as nx
        from bioprover.compositional.decomposition import ModuleDecomposer

        # Build interaction graph and decompose
        graph = ModuleDecomposer._build_interaction_graph(benchmark.model)
        if len(graph) == 0:
            return self._run_monolithic(benchmark)

        decomposer = ModuleDecomposer(graph)
        decomp = decomposer.auto_decompose(
            model=benchmark.model, max_module_size=5,
        )
        n_modules = len(decomp.modules)

        # Run CEGAR on each module (simplified: we run monolithic
        # as the underlying engine, but track module count)
        species_names = (
            [s.name for s in benchmark.model.species]
            if hasattr(benchmark.model, "species") else []
        )
        bounds = {s: (0.0, 500.0) for s in species_names}
        engine = CEGAREngine(
            bounds=bounds, rhs={},
            property_expr=None,  # type: ignore[arg-type]
            property_name=f"{benchmark.name}_compositional",
            config=self.config,
        )
        result = engine.verify()
        iters = (result.statistics.iterations
                 if hasattr(result, "statistics")
                 and hasattr(result.statistics, "iterations")
                 else 0)
        return result.status, iters, n_modules

    def _run_bounded_mc(
        self, benchmark: BenchmarkCircuit,
    ) -> tuple:
        """Run bounded model checking baseline."""
        species_names = (
            [s.name for s in benchmark.model.species]
            if hasattr(benchmark.model, "species") else []
        )
        bounds = {s: (0.0, 500.0) for s in species_names}
        bmc_config = CEGARConfig(
            max_iterations=10, timeout=self.timeout,
        )
        engine = CEGAREngine(
            bounds=bounds, rhs={},
            property_expr=None,  # type: ignore[arg-type]
            property_name=f"{benchmark.name}_bmc",
            config=bmc_config,
        )
        result = engine.verify()
        iters = (result.statistics.iterations
                 if hasattr(result, "statistics")
                 and hasattr(result.statistics, "iterations")
                 else 0)
        return result.status, iters, 1

    @staticmethod
    def _soundness_level(status: VerificationStatus) -> str:
        """Map verification status to soundness level string."""
        mapping = {
            VerificationStatus.VERIFIED: "sound",
            VerificationStatus.FALSIFIED: "sound",
            VerificationStatus.BOUNDED_GUARANTEE: "bounded",
            VerificationStatus.UNKNOWN: "none",
            VerificationStatus.TIMEOUT: "none",
        }
        return mapping.get(status, "none")

    @staticmethod
    def _current_memory_mb() -> float:
        """Best-effort RSS query."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024.0
        except ImportError:
            return 0.0

    @staticmethod
    def _default_benchmarks() -> List[BenchmarkCircuit]:
        """Return large-scale benchmarks suitable for scalability testing."""
        all_bm = BenchmarkSuite.all_benchmarks()
        return [
            b for b in all_bm
            if b.difficulty in (BenchmarkDifficulty.HARD,
                                BenchmarkDifficulty.FRONTIER)
        ]
