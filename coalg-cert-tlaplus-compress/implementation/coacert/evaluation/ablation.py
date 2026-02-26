"""
Ablation study framework for CoaCert-TLA.

Systematically disables individual components to measure their
contribution to overall compression and performance. This addresses
the review critique: "No ablation studies of algorithmic components."

Components that can be ablated:
1. Stuttering equivalence (stutter monad T)
2. Fairness constraints (Streett acceptance pairs)
3. L*-style learning (replace with naive partition refinement)
4. W-method conformance (use only BFS systematic)
5. Merkle tree witnesses (use flat hash lists)
6. Counterexample minimization (use raw counterexamples)
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AblationComponent(Enum):
    """Components that can be ablated (disabled)."""
    STUTTERING = auto()
    FAIRNESS = auto()
    LSTAR_LEARNING = auto()
    W_METHOD = auto()
    MERKLE_WITNESS = auto()
    COUNTEREXAMPLE_MINIMIZATION = auto()
    ADAPTIVE_DEPTH = auto()
    SYMMETRY_BREAKING = auto()


@dataclass
class AblationResult:
    """Result of running with one component ablated."""

    component: AblationComponent
    enabled: bool
    original_states: int = 0
    quotient_states: int = 0
    compression_ratio: float = 0.0
    total_time_seconds: float = 0.0
    learning_rounds: int = 0
    membership_queries: int = 0
    equivalence_queries: int = 0
    witness_size_bytes: int = 0
    witness_verified: bool = False
    properties_preserved: int = 0
    error: str = ""

    @property
    def compression_improvement(self) -> float:
        """How much compression improves vs no compression."""
        if self.original_states == 0:
            return 0.0
        return 1.0 - (self.quotient_states / self.original_states)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component.name,
            "enabled": self.enabled,
            "original_states": self.original_states,
            "quotient_states": self.quotient_states,
            "compression_ratio": self.compression_ratio,
            "compression_improvement_pct": self.compression_improvement * 100,
            "total_time_seconds": self.total_time_seconds,
            "learning_rounds": self.learning_rounds,
            "membership_queries": self.membership_queries,
            "equivalence_queries": self.equivalence_queries,
            "witness_size_bytes": self.witness_size_bytes,
            "witness_verified": self.witness_verified,
            "properties_preserved": self.properties_preserved,
            "error": self.error,
        }


@dataclass
class AblationStudyResult:
    """Complete ablation study results."""

    benchmark_name: str = ""
    baseline: Optional[AblationResult] = None
    ablations: List[AblationResult] = field(default_factory=list)
    total_time_seconds: float = 0.0

    def component_contribution(
        self, component: AblationComponent
    ) -> Optional[Dict[str, float]]:
        """Compute the contribution of a component.

        Contribution = (metric_with - metric_without) / metric_with
        A positive contribution means the component helps.
        """
        if self.baseline is None:
            return None

        for abl in self.ablations:
            if abl.component == component and not abl.enabled:
                baseline = self.baseline
                return {
                    "compression_contribution": (
                        (baseline.compression_improvement - abl.compression_improvement)
                        / max(baseline.compression_improvement, 0.001)
                    ),
                    "time_overhead": (
                        (baseline.total_time_seconds - abl.total_time_seconds)
                        / max(baseline.total_time_seconds, 0.001)
                    ),
                    "query_reduction": (
                        (abl.membership_queries - baseline.membership_queries)
                        / max(baseline.membership_queries, 1)
                    ),
                }
        return None

    def to_dict(self) -> Dict[str, Any]:
        contributions = {}
        for comp in AblationComponent:
            c = self.component_contribution(comp)
            if c is not None:
                contributions[comp.name] = c

        return {
            "benchmark": self.benchmark_name,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "ablations": [a.to_dict() for a in self.ablations],
            "contributions": contributions,
            "total_time_seconds": self.total_time_seconds,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_table(self) -> str:
        """Generate a text summary table."""
        lines = [
            f"Ablation Study: {self.benchmark_name}",
            "=" * 80,
        ]

        if self.baseline:
            lines.append(
                f"Baseline: {self.baseline.original_states} → "
                f"{self.baseline.quotient_states} states "
                f"({self.baseline.compression_improvement*100:.1f}% reduction) "
                f"in {self.baseline.total_time_seconds:.2f}s"
            )

        lines.append("-" * 80)
        hdr = (
            f"{'Component':<30} {'States':>8} {'Compress':>10} "
            f"{'Time':>8} {'Queries':>8} {'Δ Compress':>12}"
        )
        lines.append(hdr)
        lines.append("-" * 80)

        for abl in self.ablations:
            delta = ""
            if self.baseline:
                d = abl.compression_improvement - self.baseline.compression_improvement
                delta = f"{d*100:+.1f}%"
            lines.append(
                f"−{abl.component.name:<29} {abl.quotient_states:>8} "
                f"{abl.compression_improvement*100:>9.1f}% "
                f"{abl.total_time_seconds:>7.2f}s "
                f"{abl.membership_queries:>8} "
                f"{delta:>12}"
            )

        lines.append("-" * 80)
        return "\n".join(lines)


class AblationRunner:
    """Run ablation studies for CoaCert-TLA.

    Takes a pipeline runner callable and systematically disables
    components to measure their contribution.
    """

    def __init__(
        self,
        pipeline_runner: Callable[[Dict[str, Any]], AblationResult],
    ) -> None:
        """
        Parameters
        ----------
        pipeline_runner : callable
            A function that takes a config dict and returns an AblationResult.
            The config dict includes an "ablations" key listing disabled components.
        """
        self._runner = pipeline_runner

    def run_study(
        self,
        benchmark_name: str,
        base_config: Dict[str, Any],
        components_to_ablate: Optional[List[AblationComponent]] = None,
    ) -> AblationStudyResult:
        """Run a complete ablation study.

        Parameters
        ----------
        benchmark_name : str
            Name of the benchmark.
        base_config : dict
            Base pipeline configuration.
        components_to_ablate : list, optional
            Components to ablate. If None, ablates all components.
        """
        if components_to_ablate is None:
            components_to_ablate = list(AblationComponent)

        t0 = time.monotonic()
        study = AblationStudyResult(benchmark_name=benchmark_name)

        # Run baseline (all components enabled)
        logger.info("Running baseline for %s", benchmark_name)
        baseline_config = copy.deepcopy(base_config)
        baseline_config["ablations"] = []
        try:
            study.baseline = self._runner(baseline_config)
            study.baseline.enabled = True
        except Exception as e:
            logger.error("Baseline failed: %s", e)
            study.baseline = AblationResult(
                component=AblationComponent.STUTTERING,
                enabled=True,
                error=str(e),
            )

        # Run with each component ablated
        for comp in components_to_ablate:
            logger.info("Ablating %s for %s", comp.name, benchmark_name)
            config = copy.deepcopy(base_config)
            config["ablations"] = [comp.name]
            try:
                result = self._runner(config)
                result.component = comp
                result.enabled = False
                study.ablations.append(result)
            except Exception as e:
                logger.error("Ablation %s failed: %s", comp.name, e)
                study.ablations.append(AblationResult(
                    component=comp,
                    enabled=False,
                    error=str(e),
                ))

        study.total_time_seconds = time.monotonic() - t0
        return study


class AblationStudy:
    """High-level ablation study that systematically disables components.

    Wraps ``AblationRunner`` with a predefined component list and
    provides convenient per-component impact analysis.

    Components ablated:
        - STUTTERING: stuttering closure
        - SYMMETRY_BREAKING: symmetry detection
        - FAIRNESS: Streett acceptance / fairness constraints
        - MERKLE_WITNESS: Bloom filter / Merkle witness
        - ADAPTIVE_DEPTH: incremental deepening
    """

    STANDARD_COMPONENTS = [
        AblationComponent.STUTTERING,
        AblationComponent.SYMMETRY_BREAKING,
        AblationComponent.FAIRNESS,
        AblationComponent.MERKLE_WITNESS,
        AblationComponent.ADAPTIVE_DEPTH,
    ]

    def __init__(
        self,
        pipeline_runner: Callable[[Dict[str, Any]], AblationResult],
        components: Optional[List[AblationComponent]] = None,
    ) -> None:
        self._runner = AblationRunner(pipeline_runner)
        self._components = components or self.STANDARD_COMPONENTS

    def run(
        self,
        benchmark_name: str,
        base_config: Dict[str, Any],
    ) -> AblationStudyResult:
        """Run the ablation study and return results with per-component impact."""
        return self._runner.run_study(
            benchmark_name=benchmark_name,
            base_config=base_config,
            components_to_ablate=self._components,
        )

    @staticmethod
    def compute_impact(study: AblationStudyResult) -> Dict[str, Dict[str, float]]:
        """Compute per-component impact from an ablation study result.

        Returns a dict mapping component name to impact metrics:
            - compression_delta: change in compression improvement (negative = component helps)
            - time_delta_seconds: change in time (positive = component costs time)
            - memory_delta_pct: change in witness size
            - correctness_preserved: whether correctness was maintained
        """
        impacts: Dict[str, Dict[str, float]] = {}
        if study.baseline is None:
            return impacts

        bl = study.baseline
        for abl in study.ablations:
            if abl.enabled:
                continue
            comp_name = abl.component.name
            impacts[comp_name] = {
                "compression_delta": (
                    bl.compression_improvement - abl.compression_improvement
                ),
                "time_delta_seconds": (
                    abl.total_time_seconds - bl.total_time_seconds
                ),
                "witness_size_delta": (
                    abl.witness_size_bytes - bl.witness_size_bytes
                ),
                "correctness_preserved": float(abl.witness_verified),
                "queries_delta": (
                    abl.membership_queries - bl.membership_queries
                ),
            }
        return impacts


@dataclass
class ScalabilityDataPoint:
    """Single data point in a scalability experiment."""
    parameter_value: int  # e.g., number of processes, state space size
    original_states: int = 0
    quotient_states: int = 0
    compression_ratio: float = 0.0
    total_time_seconds: float = 0.0
    memory_mb: float = 0.0
    learning_rounds: int = 0
    membership_queries: int = 0
    timed_out: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter_value,
            "original_states": self.original_states,
            "quotient_states": self.quotient_states,
            "compression_ratio": self.compression_ratio,
            "time_seconds": self.total_time_seconds,
            "memory_mb": self.memory_mb,
            "learning_rounds": self.learning_rounds,
            "membership_queries": self.membership_queries,
            "timed_out": self.timed_out,
            "error": self.error,
        }


@dataclass
class ScalabilityResult:
    """Complete scalability analysis results."""

    benchmark_name: str = ""
    parameter_name: str = ""
    data_points: List[ScalabilityDataPoint] = field(default_factory=list)
    time_complexity_estimate: str = ""
    space_complexity_estimate: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "parameter": self.parameter_name,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "time_complexity": self.time_complexity_estimate,
            "space_complexity": self.space_complexity_estimate,
        }

    def estimate_complexity(self) -> None:
        """Estimate time and space complexity from data points."""
        valid = [dp for dp in self.data_points if not dp.timed_out and not dp.error]
        if len(valid) < 3:
            self.time_complexity_estimate = "insufficient data"
            self.space_complexity_estimate = "insufficient data"
            return

        import math as _math

        # Fit log(time) vs log(parameter) for polynomial complexity
        xs = [_math.log(dp.parameter_value) for dp in valid if dp.parameter_value > 0]
        ys_time = [_math.log(max(dp.total_time_seconds, 0.001)) for dp in valid if dp.parameter_value > 0]

        if len(xs) >= 2:
            slope = self._linear_regression_slope(xs, ys_time)
            if slope is not None:
                self.time_complexity_estimate = f"O(n^{slope:.1f})"

        # Fit log(states) vs log(parameter) for space complexity
        ys_space = [_math.log(max(dp.original_states, 1)) for dp in valid if dp.parameter_value > 0]
        if len(xs) >= 2:
            slope = self._linear_regression_slope(xs, ys_space)
            if slope is not None:
                self.space_complexity_estimate = f"O(n^{slope:.1f})"

    def _linear_regression_slope(
        self, xs: List[float], ys: List[float]
    ) -> Optional[float]:
        """Simple linear regression for slope estimation."""
        n = min(len(xs), len(ys))
        if n < 2:
            return None
        xs = xs[:n]
        ys = ys[:n]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
        den = sum((xs[i] - mean_x) ** 2 for i in range(n))
        if abs(den) < 1e-10:
            return None
        return num / den


@dataclass
class BaselineComparison:
    """Comparison between CoaCert and a baseline algorithm."""

    benchmark_name: str = ""
    coacert_states: int = 0
    coacert_time: float = 0.0
    baseline_name: str = ""
    baseline_states: int = 0
    baseline_time: float = 0.0
    states_match: bool = False
    speedup: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "coacert_states": self.coacert_states,
            "coacert_time": self.coacert_time,
            "baseline_name": self.baseline_name,
            "baseline_states": self.baseline_states,
            "baseline_time": self.baseline_time,
            "states_match": self.states_match,
            "speedup": self.speedup,
        }


@dataclass
class ComparisonSuiteResult:
    """Results of comparing CoaCert against baseline algorithms."""

    comparisons: List[BaselineComparison] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "avg_speedup": (
                sum(c.speedup for c in self.comparisons) / max(len(self.comparisons), 1)
            ),
            "all_match": all(c.states_match for c in self.comparisons),
        }

    def summary_table(self) -> str:
        lines = [
            "CoaCert vs Baseline Comparison",
            "=" * 80,
        ]
        hdr = (
            f"{'Benchmark':<25} {'CoaCert':>10} {'Baseline':>10} "
            f"{'Match':>6} {'Speedup':>8}"
        )
        lines.append(hdr)
        lines.append("-" * 80)
        for c in self.comparisons:
            lines.append(
                f"{c.benchmark_name:<25} "
                f"{c.coacert_states:>10} {c.baseline_states:>10} "
                f"{'✓' if c.states_match else '✗':>6} "
                f"{c.speedup:>7.2f}×"
            )
        lines.append("-" * 80)
        return "\n".join(lines)
