"""Benchmark runner for CollusionProof evaluation.

Orchestrates execution of evaluation scenarios, collects results, and
computes aggregate metrics such as accuracy, power, and type-I error rate.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from collusion_proof.evaluation.scenarios import (
    ScenarioSpec,
    generate_scenario_prices,
    get_all_scenarios,
    get_collusive_scenarios,
    get_competitive_scenarios,
)

logger = logging.getLogger("collusion_proof.evaluation.benchmark")


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of a single scenario evaluation."""

    scenario_id: str
    expected_verdict: str
    actual_verdict: str
    correct: bool
    confidence: float
    runtime_seconds: float
    collusion_premium: Optional[float] = None
    tier_results: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BenchmarkSummary:
    """Aggregate metrics over many scenario evaluations."""

    mode: str
    total: int
    correct: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    type_i_error: float
    type_ii_error: float
    power: float
    mean_runtime: float
    results: List[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkSummary":
        results_raw = d.pop("results", [])
        summary = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        summary.results = [BenchmarkResult.from_dict(r) for r in results_raw]
        return summary


# ---------------------------------------------------------------------------
# Detection stub — operates on raw price matrices
# ---------------------------------------------------------------------------

def _simple_detection(
    prices: np.ndarray,
    nash_price: float,
    monopoly_price: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Lightweight price-level and correlation detector.

    This is used as the built-in detection engine when no external
    ``CompositeTest`` is available.  It checks whether the converged
    mean price is significantly above Nash using a bootstrap test and
    whether player prices are highly correlated.
    """
    n_rounds, n_players = prices.shape
    # Use last 40% of rounds for steady-state analysis
    tail_start = int(n_rounds * 0.6)
    tail = prices[tail_start:]
    mean_prices = tail.mean(axis=0)
    overall_mean = float(np.mean(mean_prices))

    price_gap = monopoly_price - nash_price
    if price_gap <= 0:
        collusion_index = 0.0
    else:
        collusion_index = max(0.0, min((overall_mean - nash_price) / price_gap, 1.0))

    # Bootstrap test of H0: mean price ≤ nash_price
    rng = np.random.RandomState(0)
    flat_tail = tail.ravel()
    n_boot = 2000
    boot_means = np.empty(n_boot)
    n = len(flat_tail)
    for i in range(n_boot):
        boot_means[i] = np.mean(flat_tail[rng.randint(0, n, size=n)])
    p_value_price = float(np.mean(boot_means <= nash_price))

    # Correlation test (pairwise mean)
    if n_players >= 2:
        corr_matrix = np.corrcoef(tail.T)
        upper = corr_matrix[np.triu_indices(n_players, k=1)]
        mean_corr = float(np.mean(upper)) if len(upper) > 0 else 0.0
    else:
        mean_corr = 0.0

    # Variance-based test
    price_variance = float(np.mean(np.var(tail, axis=0)))

    # Decision logic
    tier_results: Dict[str, Any] = {
        "price_level": {
            "mean_price": overall_mean,
            "collusion_index": collusion_index,
            "p_value": p_value_price,
            "reject": p_value_price < alpha,
        },
        "correlation": {
            "mean_correlation": mean_corr,
            "high_correlation": mean_corr > 0.7,
        },
        "variance": {
            "price_variance": price_variance,
            "low_variance": price_variance < 0.05,
        },
    }

    # Determine verdict
    collusion_signals = 0
    if collusion_index > 0.3 and p_value_price < alpha:
        collusion_signals += 1
    if mean_corr > 0.7:
        collusion_signals += 1
    if price_variance < 0.05 and collusion_index > 0.2:
        collusion_signals += 1

    if collusion_signals >= 2:
        verdict = "collusive"
        confidence = min(0.5 + 0.2 * collusion_signals + collusion_index * 0.3, 0.99)
    elif collusion_index > 0.5:
        verdict = "collusive"
        confidence = 0.5 + collusion_index * 0.3
    else:
        verdict = "competitive"
        confidence = max(0.5 + (1.0 - collusion_index) * 0.3, 0.5)

    return {
        "verdict": verdict,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "collusion_premium": float(max(overall_mean - nash_price, 0.0)),
        "collusion_index": collusion_index,
        "tier_results": tier_results,
    }


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Run evaluation benchmarks against the detection pipeline."""

    def __init__(
        self,
        alpha: float = 0.05,
        bootstrap_samples: int = 10_000,
        verbose: bool = True,
        n_jobs: int = 1,
        detection_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.verbose = verbose
        self.n_jobs = n_jobs
        self._detection_fn = detection_fn or _simple_detection

    # -- public entry points -------------------------------------------------

    def run_smoke(self, seed: Optional[int] = 42) -> BenchmarkSummary:
        """Quick smoke test with a small subset of scenarios (6 total)."""
        collusive = get_collusive_scenarios()[:3]
        competitive = get_competitive_scenarios()[:3]
        scenarios = collusive + competitive
        return self._run_scenarios(scenarios, seed=seed, n_repeats=1, mode="smoke")

    def run_standard(self, seed: Optional[int] = 42) -> BenchmarkSummary:
        """Standard evaluation with all 30 scenarios, single run."""
        scenarios = get_all_scenarios()
        return self._run_scenarios(scenarios, seed=seed, n_repeats=1, mode="standard")

    def run_full(
        self, seed: Optional[int] = 42, n_repeats: int = 10,
    ) -> BenchmarkSummary:
        """Full evaluation with repeated runs for statistical power."""
        scenarios = get_all_scenarios()
        return self._run_scenarios(scenarios, seed=seed, n_repeats=n_repeats, mode="full")

    def run_scenario(
        self, scenario: ScenarioSpec, seed: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run detection on a single scenario and return the result."""
        if self.verbose:
            logger.info("Running scenario %s (%s)", scenario.scenario_id, scenario.name)

        t0 = time.perf_counter()
        prices = generate_scenario_prices(scenario, seed=seed)
        detection = self._detection_fn(
            prices,
            nash_price=scenario.nash_price,
            monopoly_price=scenario.monopoly_price,
            alpha=self.alpha,
        )
        elapsed = time.perf_counter() - t0

        actual = detection.get("verdict", "competitive")
        expected = scenario.expected_verdict

        return BenchmarkResult(
            scenario_id=scenario.scenario_id,
            expected_verdict=expected,
            actual_verdict=actual,
            correct=(actual == expected),
            confidence=detection.get("confidence", 0.5),
            runtime_seconds=elapsed,
            collusion_premium=detection.get("collusion_premium"),
            tier_results=detection.get("tier_results", {}),
            details=detection,
        )

    # -- internal helpers ----------------------------------------------------

    def _run_scenarios(
        self,
        scenarios: List[ScenarioSpec],
        seed: Optional[int],
        n_repeats: int = 1,
        mode: str = "standard",
    ) -> BenchmarkSummary:
        """Run a list of scenarios (possibly repeated) and compute summary."""
        all_results: List[BenchmarkResult] = []
        rng = np.random.RandomState(seed)

        total_runs = len(scenarios) * n_repeats
        if self.verbose:
            logger.info(
                "Starting %s benchmark: %d scenarios × %d repeats = %d runs",
                mode, len(scenarios), n_repeats, total_runs,
            )

        for rep in range(n_repeats):
            for i, scenario in enumerate(scenarios):
                run_seed = int(rng.randint(0, 2**31))
                result = self.run_scenario(scenario, seed=run_seed)
                all_results.append(result)

                if self.verbose:
                    idx = rep * len(scenarios) + i + 1
                    mark = "✓" if result.correct else "✗"
                    logger.info(
                        "  [%d/%d] %s %s  expected=%s actual=%s (%.2fs)",
                        idx, total_runs, mark, scenario.scenario_id,
                        result.expected_verdict, result.actual_verdict,
                        result.runtime_seconds,
                    )

        return self._compute_summary(all_results, mode)

    def _compute_summary(
        self, results: List[BenchmarkResult], mode: str,
    ) -> BenchmarkSummary:
        """Compute aggregate metrics from a list of results."""
        total = len(results)
        if total == 0:
            return BenchmarkSummary(
                mode=mode, total=0, correct=0, accuracy=0.0,
                precision=0.0, recall=0.0, f1=0.0,
                type_i_error=0.0, type_ii_error=0.0,
                power=0.0, mean_runtime=0.0, results=[],
            )

        correct = sum(1 for r in results if r.correct)

        # True / false positives / negatives (positive = collusive)
        tp = sum(
            1 for r in results
            if r.expected_verdict == "collusive" and r.actual_verdict == "collusive"
        )
        fp = sum(
            1 for r in results
            if r.expected_verdict != "collusive" and r.actual_verdict == "collusive"
        )
        fn = sum(
            1 for r in results
            if r.expected_verdict == "collusive" and r.actual_verdict != "collusive"
        )

        n_competitive = sum(1 for r in results if r.expected_verdict != "collusive")
        n_collusive = sum(1 for r in results if r.expected_verdict == "collusive")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        type_i = fp / n_competitive if n_competitive > 0 else 0.0
        type_ii = fn / n_collusive if n_collusive > 0 else 0.0

        mean_runtime = float(np.mean([r.runtime_seconds for r in results]))

        summary = BenchmarkSummary(
            mode=mode,
            total=total,
            correct=correct,
            accuracy=correct / total,
            precision=precision,
            recall=recall,
            f1=f1,
            type_i_error=type_i,
            type_ii_error=type_ii,
            power=1.0 - type_ii,
            mean_runtime=mean_runtime,
            results=results,
        )

        if self.verbose:
            self.print_summary(summary)

        return summary

    # -- I/O -----------------------------------------------------------------

    def save_results(self, summary: BenchmarkSummary, path: str) -> None:
        """Persist summary to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        if self.verbose:
            logger.info("Results saved to %s", path)

    def load_results(self, path: str) -> BenchmarkSummary:
        """Load a summary from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return BenchmarkSummary.from_dict(data)

    # -- display -------------------------------------------------------------

    def print_summary(self, summary: BenchmarkSummary) -> None:
        """Print a formatted summary to the logger."""
        lines = [
            "",
            f"{'=' * 50}",
            f"  Benchmark Summary ({summary.mode})",
            f"{'=' * 50}",
            f"  Total scenarios : {summary.total}",
            f"  Correct         : {summary.correct}",
            f"  Accuracy        : {summary.accuracy:.2%}",
            f"  Precision       : {summary.precision:.2%}",
            f"  Recall          : {summary.recall:.2%}",
            f"  F1 Score        : {summary.f1:.2%}",
            f"  Type I error    : {summary.type_i_error:.2%}",
            f"  Type II error   : {summary.type_ii_error:.2%}",
            f"  Power           : {summary.power:.2%}",
            f"  Mean runtime    : {summary.mean_runtime:.3f}s",
            f"{'=' * 50}",
        ]

        # Per-category breakdown
        categories: Dict[str, List[BenchmarkResult]] = {}
        for r in summary.results:
            cat = r.scenario_id.split("_")[0]
            categories.setdefault(cat, []).append(r)

        for cat, cat_results in sorted(categories.items()):
            cat_correct = sum(1 for r in cat_results if r.correct)
            cat_total = len(cat_results)
            lines.append(
                f"  {cat:>8s}: {cat_correct}/{cat_total} correct "
                f"({cat_correct / cat_total:.0%})"
            )

        lines.append(f"{'=' * 50}")
        logger.info("\n".join(lines))
