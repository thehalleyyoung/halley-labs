"""Ablation experiment infrastructure for BioProver AI components.

Runs verification with different AI configurations and compares
iterations to converge, wall-clock time, and predicate counts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration enum
# ---------------------------------------------------------------------------


class AblationConfig(Enum):
    """Available ablation configurations."""
    NO_AI = auto()                 # Pure CEGAR, no AI guidance
    AI_STRUCTURAL = auto()         # AI with structural fallback
    AI_MONOTONICITY = auto()       # AI with monotonicity fallback
    AI_ALL_STRATEGIES = auto()     # AI with all strategies enabled


ABLATION_CONFIG_NAMES = {
    AblationConfig.NO_AI: "no_ai",
    AblationConfig.AI_STRUCTURAL: "ai_structural",
    AblationConfig.AI_MONOTONICITY: "ai_monotonicity",
    AblationConfig.AI_ALL_STRATEGIES: "ai_all_strategies",
}


# ---------------------------------------------------------------------------
# Result data
# ---------------------------------------------------------------------------


@dataclass
class AblationRunResult:
    """Result of a single ablation configuration run on one circuit."""

    config: AblationConfig
    config_name: str
    circuit_name: str
    iterations: int = 0
    wall_clock_seconds: float = 0.0
    n_predicates: int = 0
    converged: bool = False
    status: str = "UNKNOWN"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config_name,
            "circuit": self.circuit_name,
            "iterations": self.iterations,
            "wall_clock_s": round(self.wall_clock_seconds, 4),
            "n_predicates": self.n_predicates,
            "converged": self.converged,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class AblationSummary:
    """Aggregated comparison across configurations."""

    results: List[AblationRunResult] = field(default_factory=list)

    def by_config(self) -> Dict[str, List[AblationRunResult]]:
        grouped: Dict[str, List[AblationRunResult]] = {}
        for r in self.results:
            grouped.setdefault(r.config_name, []).append(r)
        return grouped

    def comparison_table(self) -> Dict[str, Dict[str, Any]]:
        """Produce a comparison table keyed by configuration name."""
        table: Dict[str, Dict[str, Any]] = {}
        for cfg_name, runs in self.by_config().items():
            times = [r.wall_clock_seconds for r in runs]
            iters = [r.iterations for r in runs]
            preds = [r.n_predicates for r in runs]
            converged = sum(1 for r in runs if r.converged)
            table[cfg_name] = {
                "n_circuits": len(runs),
                "converged": converged,
                "converged_frac": converged / max(len(runs), 1),
                "mean_time_s": float(np.mean(times)),
                "median_time_s": float(np.median(times)),
                "mean_iterations": float(np.mean(iters)),
                "median_iterations": float(np.median(iters)),
                "mean_predicates": float(np.mean(preds)),
            }
        return table

    def speedup_over_baseline(
        self, baseline: str = "no_ai",
    ) -> Dict[str, float]:
        """Compute mean speedup of each config relative to baseline."""
        by_circuit: Dict[str, Dict[str, float]] = {}
        for r in self.results:
            by_circuit.setdefault(r.circuit_name, {})[r.config_name] = (
                r.wall_clock_seconds
            )
        speedups: Dict[str, List[float]] = {}
        for circuit, configs in by_circuit.items():
            base_time = configs.get(baseline)
            if base_time is None or base_time <= 0:
                continue
            for cfg, t in configs.items():
                if cfg == baseline:
                    continue
                speedups.setdefault(cfg, []).append(base_time / max(t, 1e-8))
        return {
            cfg: float(np.mean(ratios)) for cfg, ratios in speedups.items()
        }


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------


class AblationRunner:
    """Runs verification across ablation configurations.

    Parameters
    ----------
    engine_factory
        ``(circuit, config_name: str) -> engine`` callable.
        The engine must expose ``.verify() -> result`` and ``.statistics``.
    configs
        Which configurations to test.  Defaults to all.
    """

    def __init__(
        self,
        engine_factory: Callable[..., Any],
        configs: Optional[List[AblationConfig]] = None,
    ) -> None:
        self._factory = engine_factory
        self._configs = configs or list(AblationConfig)

    def run(
        self,
        circuits: List[Any],
    ) -> AblationSummary:
        """Run all configurations on all circuits."""
        summary = AblationSummary()
        for circuit in circuits:
            circuit_name = getattr(circuit, "name", str(circuit))
            for cfg in self._configs:
                cfg_name = ABLATION_CONFIG_NAMES[cfg]
                logger.info(
                    "Ablation: circuit=%s config=%s", circuit_name, cfg_name,
                )
                result = self._run_single(circuit, circuit_name, cfg, cfg_name)
                summary.results.append(result)
        return summary

    def _run_single(
        self,
        circuit: Any,
        circuit_name: str,
        cfg: AblationConfig,
        cfg_name: str,
    ) -> AblationRunResult:
        try:
            engine = self._factory(circuit, cfg_name)
            t0 = time.monotonic()
            vresult = engine.verify()
            elapsed = time.monotonic() - t0

            stats = engine.statistics
            return AblationRunResult(
                config=cfg,
                config_name=cfg_name,
                circuit_name=circuit_name,
                iterations=getattr(stats, "iterations", 0),
                wall_clock_seconds=elapsed,
                n_predicates=getattr(stats, "peak_predicates", 0),
                converged=getattr(vresult, "is_verified", False),
                status=getattr(vresult.status, "name", "UNKNOWN")
                if hasattr(vresult, "status") else "UNKNOWN",
            )
        except Exception as exc:
            logger.warning(
                "Ablation error: circuit=%s config=%s error=%s",
                circuit_name, cfg_name, exc,
            )
            return AblationRunResult(
                config=cfg,
                config_name=cfg_name,
                circuit_name=circuit_name,
                status="ERROR",
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_ablation_report(summary: AblationSummary) -> str:
    """Produce a human-readable comparison table from ablation results."""
    table = summary.comparison_table()
    speedups = summary.speedup_over_baseline()

    lines = [
        "=" * 72,
        "BioProver Ablation Study Report",
        "=" * 72,
        "",
        f"{'Config':<24} {'Circuits':>8} {'Conv%':>7} {'MeanTime':>10} "
        f"{'MeanIter':>10} {'MeanPreds':>10} {'Speedup':>8}",
        "-" * 72,
    ]
    for cfg_name, stats in table.items():
        speedup = speedups.get(cfg_name, 1.0)
        if cfg_name == "no_ai":
            speedup_str = "(base)"
        else:
            speedup_str = f"{speedup:.2f}x"
        lines.append(
            f"{cfg_name:<24} {stats['n_circuits']:>8} "
            f"{stats['converged_frac']:>6.1%} "
            f"{stats['mean_time_s']:>10.3f} "
            f"{stats['mean_iterations']:>10.1f} "
            f"{stats['mean_predicates']:>10.1f} "
            f"{speedup_str:>8}"
        )
    lines.extend(["", "=" * 72])
    return "\n".join(lines)
