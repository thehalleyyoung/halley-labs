"""Parametric sweep engine for systematic evaluation.

Provides grid, random, and adaptive sweep strategies over detection
parameter spaces, along with analysis utilities for finding optimal
configurations and estimating parameter importance.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("collusion_proof.evaluation.parametric_sweep")


@dataclass
class SweepResult:
    """Single evaluation point in a parameter sweep."""

    params: Dict[str, float]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepResult":
        return cls(params=d["params"], metrics=d["metrics"])


class ParametricSweep:
    """Systematic parameter sweep for evaluation."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Grid sweep
    # ------------------------------------------------------------------

    def grid_sweep(
        self,
        param_grid: Dict[str, List[float]],
        objective: Callable[..., Dict[str, float]],
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> List[SweepResult]:
        """Full grid sweep over all parameter combinations.

        Parameters
        ----------
        param_grid : {name: [val1, val2, …]} for each swept parameter.
        objective : callable(**params) -> dict of metric values.
        fixed_params : extra constant kwargs passed to *objective*.

        Returns
        -------
        List of :class:`SweepResult`, one per grid point.
        """
        if fixed_params is None:
            fixed_params = {}

        names = sorted(param_grid.keys())
        value_lists = [param_grid[n] for n in names]
        combos = list(itertools.product(*value_lists))
        total = len(combos)

        if self.verbose:
            logger.info("Grid sweep: %d combinations over %d params", total, len(names))

        results: List[SweepResult] = []
        for idx, combo in enumerate(combos):
            params = {name: float(val) for name, val in zip(names, combo)}
            all_params = {**fixed_params, **params}
            metrics = objective(**all_params)
            results.append(SweepResult(params=params, metrics=dict(metrics)))

            if self.verbose and (idx + 1) % max(total // 10, 1) == 0:
                logger.info("  grid sweep %d/%d", idx + 1, total)

        return results

    # ------------------------------------------------------------------
    # Random sweep
    # ------------------------------------------------------------------

    def random_sweep(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objective: Callable[..., Dict[str, float]],
        n_samples: int = 100,
        fixed_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> List[SweepResult]:
        """Random parameter sampling (uniform)."""
        if fixed_params is None:
            fixed_params = {}

        rng = np.random.RandomState(seed)
        names = sorted(param_ranges.keys())

        if self.verbose:
            logger.info("Random sweep: %d samples over %d params", n_samples, len(names))

        results: List[SweepResult] = []
        for i in range(n_samples):
            params: Dict[str, float] = {}
            for name in names:
                lo, hi = param_ranges[name]
                params[name] = float(rng.uniform(lo, hi))
            all_params = {**fixed_params, **params}
            metrics = objective(**all_params)
            results.append(SweepResult(params=params, metrics=dict(metrics)))

            if self.verbose and (i + 1) % max(n_samples // 10, 1) == 0:
                logger.info("  random sweep %d/%d", i + 1, n_samples)

        return results

    # ------------------------------------------------------------------
    # Adaptive sweep
    # ------------------------------------------------------------------

    def adaptive_sweep(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        objective: Callable[..., Dict[str, float]],
        n_initial: int = 20,
        n_refine: int = 30,
        target_metric: str = "accuracy",
        minimize: bool = False,
        fixed_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> List[SweepResult]:
        """Adaptive refinement: sample densely near the best regions.

        Phase 1: ``n_initial`` random samples.
        Phase 2: ``n_refine`` samples drawn from shrunk ranges centred
        on the best points found so far.
        """
        if fixed_params is None:
            fixed_params = {}
        rng = np.random.RandomState(seed)
        names = sorted(param_ranges.keys())

        # Phase 1: initial random exploration
        initial_results = self.random_sweep(
            param_ranges, objective, n_samples=n_initial,
            fixed_params=fixed_params, seed=int(rng.randint(0, 2**31)),
        )

        # Find top-k points
        k = max(1, n_initial // 5)
        sorted_results = sorted(
            initial_results,
            key=lambda r: r.metrics.get(target_metric, 0.0),
            reverse=not minimize,
        )
        top_k = sorted_results[:k]

        # Phase 2: refine around top-k
        refine_results: List[SweepResult] = []
        samples_per_point = max(1, n_refine // k)

        for centre in top_k:
            for _ in range(samples_per_point):
                params: Dict[str, float] = {}
                for name in names:
                    lo, hi = param_ranges[name]
                    span = (hi - lo) * 0.25  # shrink to 25% of original range
                    mid = centre.params[name]
                    new_lo = max(lo, mid - span)
                    new_hi = min(hi, mid + span)
                    params[name] = float(rng.uniform(new_lo, new_hi))
                all_params = {**fixed_params, **params}
                metrics = objective(**all_params)
                refine_results.append(SweepResult(params=params, metrics=dict(metrics)))

        all_results = initial_results + refine_results

        if self.verbose:
            logger.info(
                "Adaptive sweep: %d initial + %d refine = %d total",
                len(initial_results), len(refine_results), len(all_results),
            )

        return all_results

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def analyze_results(self, results: List[SweepResult]) -> Dict[str, Any]:
        """Analyse sweep results: best params, summary stats, correlations."""
        if not results:
            return {"error": "no results"}

        metric_names = list(results[0].metrics.keys())
        param_names = list(results[0].params.keys())

        analysis: Dict[str, Any] = {
            "n_points": len(results),
            "param_names": param_names,
            "metric_names": metric_names,
            "metric_stats": {},
            "best_per_metric": {},
        }

        for metric in metric_names:
            vals = np.array([r.metrics.get(metric, np.nan) for r in results])
            valid = vals[~np.isnan(vals)]
            analysis["metric_stats"][metric] = {
                "mean": float(np.mean(valid)) if len(valid) > 0 else float("nan"),
                "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                "min": float(np.min(valid)) if len(valid) > 0 else float("nan"),
                "max": float(np.max(valid)) if len(valid) > 0 else float("nan"),
                "median": float(np.median(valid)) if len(valid) > 0 else float("nan"),
            }

            # Best point
            best = self.find_optimal(results, metric, minimize=False)
            analysis["best_per_metric"][metric] = {
                "params": best.params,
                "value": best.metrics.get(metric, float("nan")),
            }

        return analysis

    def find_optimal(
        self, results: List[SweepResult], metric: str, minimize: bool = False,
    ) -> SweepResult:
        """Find the parameter combination that optimises *metric*."""
        if not results:
            raise ValueError("No results to search")

        def key_fn(r: SweepResult) -> float:
            v = r.metrics.get(metric, float("-inf") if not minimize else float("inf"))
            return v if not minimize else -v

        return max(results, key=key_fn)

    def parameter_importance(
        self, results: List[SweepResult], metric: str,
    ) -> Dict[str, float]:
        """Estimate parameter importance via correlation analysis.

        Computes the absolute Spearman rank correlation between each
        parameter and the target metric.  Higher values indicate the
        parameter has a bigger influence on the metric.

        If scikit-learn is available and enough samples exist, a random
        forest permutation importance is used instead.
        """
        if not results:
            return {}

        param_names = sorted(results[0].params.keys())
        n = len(results)
        X = np.array([[r.params[p] for p in param_names] for r in results])
        y = np.array([r.metrics.get(metric, 0.0) for r in results])

        # Try sklearn random forest importance first
        try:
            if n >= 20:
                from sklearn.ensemble import RandomForestRegressor

                rf = RandomForestRegressor(
                    n_estimators=100, max_depth=5, random_state=42,
                )
                rf.fit(X, y)
                importances = rf.feature_importances_
                return {
                    name: float(imp)
                    for name, imp in zip(param_names, importances)
                }
        except ImportError:
            pass

        # Fallback: absolute Spearman correlation
        importance: Dict[str, float] = {}
        for j, name in enumerate(param_names):
            rank_x = _rank_data(X[:, j])
            rank_y = _rank_data(y)
            corr = _pearson_corr(rank_x, rank_y)
            importance[name] = abs(corr)

        return importance

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_results(self, results: List[SweepResult], path: str) -> None:
        """Persist sweep results to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, default=str)
        if self.verbose:
            logger.info("Sweep results saved to %s (%d points)", path, len(results))

    def load_results(self, path: str) -> List[SweepResult]:
        """Load sweep results from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return [SweepResult.from_dict(d) for d in data]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_data(arr: np.ndarray) -> np.ndarray:
    """Rank array values (average rank for ties)."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # Average ranks for ties
    unique_vals = np.unique(arr)
    for val in unique_vals:
        mask = arr == val
        if np.sum(mask) > 1:
            ranks[mask] = np.mean(ranks[mask])
    return ranks


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    if sx < 1e-30 or sy < 1e-30:
        return 0.0
    return float(np.sum((x - mx) * (y - my)) / ((n - 1) * sx * sy))
