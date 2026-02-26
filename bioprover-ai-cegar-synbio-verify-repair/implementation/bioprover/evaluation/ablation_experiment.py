"""Ablation study for AI components in BioProver.

Compares verification performance with and without:
1. AI-guided predicate selection
2. GP robustness surrogate
3. Monotonicity refinement
4. Time-scale refinement
5. Full BioProver (all components)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    "full",                  # All components enabled
    "no_ai",                 # AI-guided predicate selection disabled
    "no_surrogate",          # GP robustness surrogate disabled
    "no_monotonicity",       # Monotonicity refinement disabled
    "no_timescale",          # Time-scale refinement disabled
    "structural_only",       # Only structural refinement
]


# ---------------------------------------------------------------------------
# Result data
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    """Result of a single ablation experiment run."""

    configuration: str
    circuit_name: str
    iterations: int
    time_seconds: float
    status: str
    soundness_level: str
    predicates_tried: int
    predicates_useful: int

    @property
    def predicate_efficiency(self) -> float:
        if self.predicates_tried == 0:
            return 0.0
        return self.predicates_useful / self.predicates_tried

    def to_dict(self) -> Dict[str, Any]:
        return {
            "configuration": self.configuration,
            "circuit": self.circuit_name,
            "iterations": self.iterations,
            "time_s": round(self.time_seconds, 3),
            "status": self.status,
            "soundness": self.soundness_level,
            "predicates_tried": self.predicates_tried,
            "predicates_useful": self.predicates_useful,
            "predicate_efficiency": round(self.predicate_efficiency, 4),
        }


@dataclass
class AblationReport:
    """Aggregated report for an ablation study."""

    results: List[AblationResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Per-configuration aggregate statistics."""
        import numpy as np

        by_config: Dict[str, List[AblationResult]] = {}
        for r in self.results:
            by_config.setdefault(r.configuration, []).append(r)

        summary: Dict[str, Dict[str, Any]] = {}
        for config, runs in by_config.items():
            times = [r.time_seconds for r in runs]
            iters = [r.iterations for r in runs]
            verified = sum(1 for r in runs if r.status == "VERIFIED")
            summary[config] = {
                "n_circuits": len(runs),
                "verified": verified,
                "verified_frac": verified / max(len(runs), 1),
                "mean_time": float(np.mean(times)),
                "median_time": float(np.median(times)),
                "mean_iterations": float(np.mean(iters)),
                "mean_predicate_efficiency": float(
                    np.mean([r.predicate_efficiency for r in runs])
                ),
            }
        return summary

    def speedup_vs_baseline(self, baseline: str = "structural_only") -> Dict[str, float]:
        """Compute speedup of each configuration relative to *baseline*."""
        import numpy as np

        by_circuit: Dict[str, Dict[str, float]] = {}
        for r in self.results:
            by_circuit.setdefault(r.circuit_name, {})[r.configuration] = r.time_seconds

        speedups: Dict[str, List[float]] = {}
        for circuit, configs in by_circuit.items():
            base_time = configs.get(baseline)
            if base_time is None or base_time <= 0:
                continue
            for cfg, t in configs.items():
                if cfg == baseline:
                    continue
                speedups.setdefault(cfg, []).append(base_time / max(t, 1e-6))

        return {
            cfg: float(np.mean(ratios)) for cfg, ratios in speedups.items()
        }


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------


def run_ablation(
    benchmark_circuits: List[Any],
    cegar_engine_factory: Callable[..., Any],
    configs: Optional[List[str]] = None,
) -> AblationReport:
    """Run ablation study across all benchmarks and configurations.

    Parameters
    ----------
    benchmark_circuits
        List of benchmark objects.  Each must have ``name`` and attributes
        needed by *cegar_engine_factory*.
    cegar_engine_factory
        ``(circuit, config_name) -> CEGAREngine`` factory.  The factory
        must configure the engine according to the ablation config name.
    configs
        Which configurations to test.  Defaults to :data:`ABLATION_CONFIGS`.

    Returns
    -------
    AblationReport
    """
    if configs is None:
        configs = list(ABLATION_CONFIGS)

    report = AblationReport()

    for circuit in benchmark_circuits:
        circuit_name = getattr(circuit, "name", str(circuit))
        for config_name in configs:
            logger.info(
                "Ablation: circuit=%s config=%s", circuit_name, config_name,
            )
            try:
                engine = cegar_engine_factory(circuit, config_name)
                t0 = time.monotonic()
                result = engine.verify()
                elapsed = time.monotonic() - t0

                stats = engine.statistics
                report.results.append(AblationResult(
                    configuration=config_name,
                    circuit_name=circuit_name,
                    iterations=stats.iterations,
                    time_seconds=elapsed,
                    status=result.status.name,
                    soundness_level=(
                        result.termination_reason.name
                        if result.termination_reason else "UNKNOWN"
                    ),
                    predicates_tried=stats.peak_predicates,
                    predicates_useful=stats.peak_predicates - stats.spurious_count,
                ))
            except Exception as exc:
                logger.warning(
                    "Ablation failed: circuit=%s config=%s error=%s",
                    circuit_name, config_name, exc,
                )
                report.results.append(AblationResult(
                    configuration=config_name,
                    circuit_name=circuit_name,
                    iterations=0,
                    time_seconds=0.0,
                    status="ERROR",
                    soundness_level="NONE",
                    predicates_tried=0,
                    predicates_useful=0,
                ))

    return report
