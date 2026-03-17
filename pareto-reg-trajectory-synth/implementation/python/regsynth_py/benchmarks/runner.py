"""Benchmark runner: execute benchmarks and collect metrics.

Provides built-in greedy, random, and exhaustive solvers as baselines and
supports plugging in arbitrary solver callables.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
from typing import Any, Callable


class BenchmarkRunner:
    """Execute benchmark instances and collect quality / timing metrics."""

    def __init__(self, timeout: float = 300.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        instance: dict,
        solver_fn: Callable[[dict], list[dict]] | None = None,
    ) -> dict:
        """Run a single benchmark instance with *solver_fn*.

        If *solver_fn* is ``None`` the built-in greedy solver is used.
        """
        solver = solver_fn or self._greedy_solver
        solver_name = getattr(solver, "__name__", str(solver))

        elapsed, solutions = self._measure_time(solver, instance)

        # Enforce timeout
        status = "ok" if elapsed <= self._timeout else "timeout"

        metrics = self._compute_metrics(solutions, instance)

        return {
            "instance_id": instance.get("id", "unknown"),
            "solver": solver_name,
            "time_seconds": round(elapsed, 6),
            "solutions": solutions,
            "metrics": metrics,
            "status": status,
        }

    def run_suite(
        self,
        instances: list[dict],
        solver_fn: Callable[[dict], list[dict]] | None = None,
    ) -> list[dict]:
        """Run multiple benchmark instances and return aggregated results."""
        results: list[dict] = []
        for inst in instances:
            results.append(self.run(inst, solver_fn=solver_fn))
        return results

    def run_comparison(
        self,
        instance: dict,
        solver_fns: dict[str, Callable[[dict], list[dict]]],
    ) -> dict:
        """Run the same instance with multiple solvers and compare."""
        comparison: dict[str, dict] = {}
        for name, fn in solver_fns.items():
            result = self.run(instance, solver_fn=fn)
            result["solver"] = name
            comparison[name] = result
        return {
            "instance_id": instance.get("id", "unknown"),
            "solvers": comparison,
        }

    # ------------------------------------------------------------------
    # Built-in solvers
    # ------------------------------------------------------------------

    def _greedy_solver(self, instance: dict) -> list[dict]:
        """Greedy baseline: select obligations by cost-effectiveness.

        Returns a list of Pareto-approximate solutions by incrementally
        adding the cheapest remaining non-conflicting obligation.
        """
        obligations = list(instance.get("obligations", []))
        conflicts = instance.get("conflicts", [])
        conflict_set = self._build_conflict_set(conflicts)

        obligations.sort(key=lambda o: o.get("estimated_cost", float("inf")))

        solutions: list[dict] = []
        selected: list[str] = []
        total_cost = 0.0

        for obl in obligations:
            oid = obl["id"]
            if any(
                (min(oid, s), max(oid, s)) in conflict_set
                for s in selected
            ):
                continue
            selected.append(oid)
            total_cost += obl.get("estimated_cost", 0.0)
            n_all = max(len(obligations), 1)
            solutions.append({
                "obligations": list(selected),
                "cost": round(total_cost, 2),
                "coverage": round(len(selected) / n_all, 4),
            })

        return solutions

    def _random_solver(self, instance: dict) -> list[dict]:
        """Random baseline: randomly select obligations avoiding conflicts."""
        rng = random.Random(hash(instance.get("id", 0)))
        obligations = list(instance.get("obligations", []))
        conflicts = instance.get("conflicts", [])
        conflict_set = self._build_conflict_set(conflicts)

        rng.shuffle(obligations)

        solutions: list[dict] = []
        selected: list[str] = []
        total_cost = 0.0

        for obl in obligations:
            oid = obl["id"]
            if any(
                (min(oid, s), max(oid, s)) in conflict_set
                for s in selected
            ):
                continue
            selected.append(oid)
            total_cost += obl.get("estimated_cost", 0.0)
            n_all = max(len(obligations), 1)
            solutions.append({
                "obligations": list(selected),
                "cost": round(total_cost, 2),
                "coverage": round(len(selected) / n_all, 4),
            })

        return solutions

    def _exhaustive_solver(
        self, instance: dict, max_size: int = 15
    ) -> list[dict]:
        """Exact solver for small instances — enumerates all subsets."""
        obligations = instance.get("obligations", [])
        if len(obligations) > max_size:
            return self._greedy_solver(instance)

        conflicts = instance.get("conflicts", [])
        conflict_set = self._build_conflict_set(conflicts)
        n_all = max(len(obligations), 1)

        feasible: list[dict] = []
        for r in range(1, len(obligations) + 1):
            for combo in itertools.combinations(obligations, r):
                ids = [o["id"] for o in combo]
                if self._subset_feasible(ids, conflict_set):
                    cost = sum(o.get("estimated_cost", 0.0) for o in combo)
                    feasible.append({
                        "obligations": ids,
                        "cost": round(cost, 2),
                        "coverage": round(len(ids) / n_all, 4),
                    })

        # Keep only Pareto-optimal
        pareto: list[dict] = []
        for p in feasible:
            dominated = False
            for q in feasible:
                if q is p:
                    continue
                if q["cost"] <= p["cost"] and q["coverage"] >= p["coverage"]:
                    if q["cost"] < p["cost"] or q["coverage"] > p["coverage"]:
                        dominated = True
                        break
            if not dominated:
                pareto.append(p)
        return pareto

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, solutions: list[dict], instance: dict
    ) -> dict:
        """Compute quality metrics for a set of solutions."""
        if not solutions:
            return {
                "hypervolume": 0.0,
                "coverage": 0.0,
                "total_cost": 0.0,
                "n_solutions": 0,
                "spread": 0.0,
                "optimality_gap": None,
            }

        costs = [s.get("cost", 0.0) for s in solutions]
        coverages = [s.get("coverage", 0.0) for s in solutions]

        # Hypervolume with reference point (max_cost * 1.1, 0)
        ref_cost = max(costs) * 1.1 if costs else 1.0
        hv = self._compute_hypervolume_2d(costs, coverages, ref_cost)

        best_coverage = max(coverages) if coverages else 0.0
        total_cost = sum(costs)
        spread = self._compute_spread(costs, coverages)

        # Optimality gap relative to planted solution
        planted = instance.get("planted_solution", {})
        opt_coverage = planted.get("optimal_coverage")
        gap = None
        if opt_coverage is not None and opt_coverage > 0:
            gap = self._compute_gap(best_coverage, opt_coverage)

        # Feasibility check
        n_feasible = sum(
            1 for s in solutions if self._check_feasibility(s, instance)
        )

        return {
            "hypervolume": round(hv, 6),
            "coverage": round(best_coverage, 4),
            "total_cost": round(total_cost, 2),
            "n_solutions": len(solutions),
            "n_feasible": n_feasible,
            "spread": round(spread, 6),
            "optimality_gap": round(gap, 6) if gap is not None else None,
        }

    def _check_feasibility(self, solution: dict, instance: dict) -> bool:
        """Verify that *solution* respects conflict constraints."""
        conflicts = instance.get("conflicts", [])
        conflict_set = self._build_conflict_set(conflicts)
        ids = solution.get("obligations", [])
        return self._subset_feasible(ids, conflict_set)

    @staticmethod
    def _compute_gap(
        solution_quality: float, optimal_quality: float
    ) -> float:
        """Optimality gap as a percentage."""
        if optimal_quality == 0:
            return 0.0
        return (optimal_quality - solution_quality) / optimal_quality * 100.0

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(results: dict | list, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)

    @staticmethod
    def load_results(filepath: str) -> dict | list:
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def summary(self, results: dict) -> str:
        """Human-readable summary of run results."""
        if isinstance(results, list):
            lines = [f"Suite: {len(results)} instances"]
            for r in results:
                m = r.get("metrics", {})
                lines.append(
                    f"  {r.get('instance_id')}: "
                    f"time={r.get('time_seconds', '?')}s  "
                    f"hv={m.get('hypervolume', '?')}  "
                    f"gap={m.get('optimality_gap', '?')}"
                )
            return "\n".join(lines)

        m = results.get("metrics", {})
        return (
            f"Instance {results.get('instance_id')}\n"
            f"  Solver     : {results.get('solver')}\n"
            f"  Time       : {results.get('time_seconds')} s\n"
            f"  Status     : {results.get('status')}\n"
            f"  Solutions  : {m.get('n_solutions')}\n"
            f"  Hypervolume: {m.get('hypervolume')}\n"
            f"  Coverage   : {m.get('coverage')}\n"
            f"  Opt. Gap   : {m.get('optimality_gap')}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_time(
        fn: Callable, *args: Any
    ) -> tuple[float, Any]:
        start = time.perf_counter()
        result = fn(*args)
        elapsed = time.perf_counter() - start
        return elapsed, result

    @staticmethod
    def _build_conflict_set(
        conflicts: list[dict],
    ) -> set[tuple[str, str]]:
        """Build a set of sorted obligation-id pairs for O(1) lookup."""
        cs: set[tuple[str, str]] = set()
        for c in conflicts:
            a, b = c.get("obligation_a", ""), c.get("obligation_b", "")
            cs.add((min(a, b), max(a, b)))
        return cs

    @staticmethod
    def _subset_feasible(
        ids: list[str], conflict_set: set[tuple[str, str]]
    ) -> bool:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                if pair in conflict_set:
                    return False
        return True

    @staticmethod
    def _compute_hypervolume_2d(
        costs: list[float],
        coverages: list[float],
        ref_cost: float,
    ) -> float:
        """2-D hypervolume (minimize cost, maximize coverage).

        Reference point: (ref_cost, 0).
        """
        points = sorted(zip(costs, coverages), key=lambda p: p[0])
        # Filter dominated
        filtered: list[tuple[float, float]] = []
        best_cov = -1.0
        for c, v in reversed(points):
            if v > best_cov:
                filtered.append((c, v))
                best_cov = v
        filtered.reverse()

        hv = 0.0
        prev_cov = 0.0
        for c, v in filtered:
            width = ref_cost - c
            if width > 0 and v > prev_cov:
                hv += width * (v - prev_cov)
                prev_cov = v
        return hv

    @staticmethod
    def _compute_spread(
        costs: list[float], coverages: list[float]
    ) -> float:
        """Spread: maximum Euclidean distance between solutions."""
        if len(costs) < 2:
            return 0.0
        max_dist = 0.0
        for i in range(len(costs)):
            for j in range(i + 1, len(costs)):
                d = math.hypot(costs[i] - costs[j], coverages[i] - coverages[j])
                if d > max_dist:
                    max_dist = d
        return max_dist
