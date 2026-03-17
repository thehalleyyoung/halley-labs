"""Benchmark evaluation: compare solvers, compute metrics, significance tests.

Provides Mann-Whitney U tests, Vargha-Delaney effect sizes, ranking,
and formatted comparison tables (ASCII and LaTeX).
"""

from __future__ import annotations

import json
import math
from typing import Any


class BenchmarkEvaluator:
    """Evaluate and compare benchmark run results across solvers."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    def evaluate(self, results: list[dict]) -> dict:
        """Compute aggregate metrics over a list of run results."""
        if not results:
            return {"n_runs": 0, "metrics": {}}

        metric_keys = set()
        for r in results:
            metric_keys.update(r.get("metrics", {}).keys())

        aggregated: dict[str, dict] = {}
        for key in sorted(metric_keys):
            values = [
                r["metrics"][key]
                for r in results
                if key in r.get("metrics", {}) and r["metrics"][key] is not None
            ]
            if not values:
                continue
            fvalues = [float(v) for v in values]
            aggregated[key] = {
                "mean": _mean(fvalues),
                "std": _std(fvalues),
                "min": min(fvalues),
                "max": max(fvalues),
                "median": _median(fvalues),
                "count": len(fvalues),
            }

        return {
            "n_runs": len(results),
            "metrics": aggregated,
        }

    # ------------------------------------------------------------------
    # Solver comparison
    # ------------------------------------------------------------------

    def compare_solvers(self, results: dict[str, list[dict]]) -> dict:
        """Pairwise comparison of solvers on each metric.

        *results*: ``{solver_name: [run_result, ...]}``.
        """
        solver_names = sorted(results.keys())
        all_metrics = set()
        for runs in results.values():
            for r in runs:
                all_metrics.update(r.get("metrics", {}).keys())

        comparisons: dict[str, dict] = {}
        for metric in sorted(all_metrics):
            pairwise: list[dict] = []
            for i, a in enumerate(solver_names):
                for b in solver_names[i + 1:]:
                    scores_a = _extract_metric(results[a], metric)
                    scores_b = _extract_metric(results[b], metric)
                    if scores_a and scores_b:
                        test = self.statistical_test(scores_a, scores_b)
                        pairwise.append({
                            "solver_a": a,
                            "solver_b": b,
                            "mean_a": _mean(scores_a),
                            "mean_b": _mean(scores_b),
                            **test,
                        })
            comparisons[metric] = pairwise

        return {"comparisons": comparisons}

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def compute_ranking(
        self, results: dict[str, list[dict]], metric: str = "hypervolume"
    ) -> list[tuple[str, float, float]]:
        """Rank solvers by *metric*.  Returns [(name, mean, std)]."""
        ranking: list[tuple[str, float, float]] = []
        for name, runs in results.items():
            scores = _extract_metric(runs, metric)
            if scores:
                ranking.append((name, _mean(scores), _std(scores)))
        # Higher is better for hypervolume/coverage; sort descending
        ranking.sort(key=lambda t: t[1], reverse=True)
        return ranking

    # ------------------------------------------------------------------
    # Quality / gap analysis
    # ------------------------------------------------------------------

    def quality_profile(self, results: list[dict]) -> dict:
        """Per-instance-size quality statistics."""
        by_size: dict[int, list[dict]] = {}
        for r in results:
            # Infer size from number of obligations in solutions or config
            sols = r.get("solutions", [])
            size = max((len(s.get("obligations", [])) for s in sols), default=0)
            by_size.setdefault(size, []).append(r)

        profile: dict[int, dict] = {}
        for size, runs in sorted(by_size.items()):
            hvs = _extract_metric(runs, "hypervolume")
            covs = _extract_metric(runs, "coverage")
            profile[size] = {
                "hypervolume": {"mean": _mean(hvs), "std": _std(hvs)} if hvs else {},
                "coverage": {"mean": _mean(covs), "std": _std(covs)} if covs else {},
                "n_runs": len(runs),
            }
        return profile

    def optimality_gap_analysis(self, results: list[dict]) -> dict:
        """Analyse optimality gaps across results."""
        gaps = _extract_metric(results, "optimality_gap")
        if not gaps:
            return {"available": False}
        return {
            "available": True,
            "mean_gap": _mean(gaps),
            "std_gap": _std(gaps),
            "min_gap": min(gaps),
            "max_gap": max(gaps),
            "median_gap": _median(gaps),
            "n": len(gaps),
        }

    def feasibility_rate(self, results: list[dict]) -> dict:
        """Fraction of feasible solutions per solver."""
        by_solver: dict[str, list[dict]] = {}
        for r in results:
            name = r.get("solver", "unknown")
            by_solver.setdefault(name, []).append(r)

        rates: dict[str, float] = {}
        for name, runs in by_solver.items():
            total = 0
            feasible = 0
            for r in runs:
                m = r.get("metrics", {})
                n_sol = m.get("n_solutions", 0)
                n_feas = m.get("n_feasible", n_sol)
                total += n_sol
                feasible += n_feas
            rates[name] = feasible / max(total, 1)
        return rates

    # ------------------------------------------------------------------
    # Runtime comparison
    # ------------------------------------------------------------------

    def runtime_comparison(self, results: dict[str, list[dict]]) -> dict:
        """Compute runtime statistics per solver."""
        stats: dict[str, dict] = {}
        for name, runs in results.items():
            times = [r.get("time_seconds", 0.0) for r in runs]
            if not times:
                continue
            stats[name] = {
                "mean_time": _mean(times),
                "median_time": _median(times),
                "max_time": max(times),
                "std_time": _std(times),
                "n_runs": len(times),
            }
        return stats

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def statistical_test(
        self, scores_a: list[float], scores_b: list[float]
    ) -> dict:
        """Mann-Whitney U test (two-sided) with Vargha-Delaney effect size."""
        u_stat, p_value = _mann_whitney_u(scores_a, scores_b)
        es = self.effect_size(scores_a, scores_b)
        return {
            "u_stat": u_stat,
            "p_value": round(p_value, 8),
            "significant": p_value < 0.05,
            "effect_size": round(es, 4),
        }

    @staticmethod
    def effect_size(
        scores_a: list[float], scores_b: list[float]
    ) -> float:
        """Vargha-Delaney A measure.

        Returns probability that a random draw from *scores_a* is greater
        than a random draw from *scores_b*.
        """
        if not scores_a or not scores_b:
            return 0.5
        count = 0.0
        for a in scores_a:
            for b in scores_b:
                if a > b:
                    count += 1.0
                elif a == b:
                    count += 0.5
        return count / (len(scores_a) * len(scores_b))

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def generate_comparison_table(self, results: dict[str, list[dict]]) -> str:
        """ASCII table comparing solvers across key metrics."""
        metrics = ["hypervolume", "coverage", "optimality_gap"]
        solvers = sorted(results.keys())

        col_w = 14
        header = f"{'Solver':<20}" + "".join(f"{m:>{col_w}}" for m in metrics)
        sep = "-" * len(header)
        rows = [sep, header, sep]

        for name in solvers:
            runs = results[name]
            cells: list[str] = []
            for m in metrics:
                vals = _extract_metric(runs, m)
                if vals:
                    cells.append(f"{_mean(vals):>{col_w}.4f}")
                else:
                    cells.append(f"{'N/A':>{col_w}}")
            rows.append(f"{name:<20}" + "".join(cells))
        rows.append(sep)
        return "\n".join(rows)

    def generate_latex_table(self, results: dict[str, list[dict]]) -> str:
        """LaTeX table for paper inclusion."""
        metrics = ["hypervolume", "coverage", "optimality_gap"]
        solvers = sorted(results.keys())

        n_cols = 1 + len(metrics)
        col_spec = "l" + "r" * len(metrics)
        lines: list[str] = [
            r"\begin{table}[ht]",
            r"\centering",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            "Solver & " + " & ".join(m.replace("_", r"\_") for m in metrics) + r" \\",
            r"\midrule",
        ]

        for name in solvers:
            runs = results[name]
            cells: list[str] = []
            for m in metrics:
                vals = _extract_metric(runs, m)
                if vals:
                    mu = _mean(vals)
                    sd = _std(vals)
                    cells.append(f"${mu:.4f} \\pm {sd:.4f}$")
                else:
                    cells.append("---")
            safe_name = name.replace("_", r"\_")
            lines.append(safe_name + " & " + " & ".join(cells) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Solver comparison across benchmark metrics.}",
            r"\label{tab:solver_comparison}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Summary / serialisation
    # ------------------------------------------------------------------

    def summary(self, evaluation: dict) -> str:
        """Human-readable summary of evaluation results."""
        lines: list[str] = [f"Evaluation ({evaluation.get('n_runs', '?')} runs)"]
        for key, stats in evaluation.get("metrics", {}).items():
            mu = stats.get("mean", "?")
            sd = stats.get("std", "?")
            lines.append(f"  {key:<20}: mean={mu:.4f}  std={sd:.4f}" if isinstance(mu, float) else f"  {key:<20}: {mu}")
        return "\n".join(lines)

    @staticmethod
    def to_json(evaluation: dict) -> str:
        return json.dumps(evaluation, indent=2, default=str)


# ======================================================================
# Module-level helpers (stdlib-only statistics)
# ======================================================================

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _extract_metric(runs: list[dict], metric: str) -> list[float]:
    """Pull numeric metric values from run results."""
    out: list[float] = []
    for r in runs:
        v = r.get("metrics", {}).get(metric)
        if v is not None:
            out.append(float(v))
    return out


def _mann_whitney_u(
    x: list[float], y: list[float]
) -> tuple[float, float]:
    """Mann-Whitney U test (two-sided), stdlib-only implementation.

    Returns (U-statistic, approximate p-value).
    Uses normal approximation for n > 20; otherwise exact is impractical
    so we still use the approximation with a conservative floor.
    """
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0, 1.0

    # Compute U
    u = 0.0
    for xi in x:
        for yj in y:
            if xi > yj:
                u += 1.0
            elif xi == yj:
                u += 0.5

    # Normal approximation
    mu_u = nx * ny / 2.0
    sigma_u = math.sqrt(nx * ny * (nx + ny + 1) / 12.0)
    if sigma_u == 0:
        return u, 1.0

    z = (u - mu_u) / sigma_u
    # Two-sided p-value from standard normal CDF
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return round(u, 4), max(p, 0.0)


def _normal_cdf(z: float) -> float:
    """Approximation of the standard normal CDF (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
