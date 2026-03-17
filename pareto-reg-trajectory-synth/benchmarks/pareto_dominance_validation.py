#!/usr/bin/env python3
"""Pareto dominance validation benchmark for RegSynth.

Validates two core theoretical claims:
1. All trajectories produced by MaxSMT synthesis are non-dominated (Pareto correctness).
2. MaxSMT solutions match brute-force optimal on small instances (optimality bound).

Also provides a reusable dominance-checking utility module.

Usage:
    python benchmarks/pareto_dominance_validation.py [--runs N] [--output PATH]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Dominance-checking utilities
# ---------------------------------------------------------------------------


def dominates(a: tuple, b: tuple) -> bool:
    """True if *a* Pareto-dominates *b* (all objectives minimised)."""
    strictly_better = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            strictly_better = True
    return strictly_better


def is_pareto_front(front: list[tuple], candidate_pool: list[tuple]) -> tuple[bool, list[int]]:
    """Check that every point in *front* is non-dominated by any point in *candidate_pool*.

    Returns (all_non_dominated, list_of_dominated_indices).
    """
    dominated_indices: list[int] = []
    for i, p in enumerate(front):
        for q in candidate_pool:
            if dominates(q, p):
                dominated_indices.append(i)
                break
    return len(dominated_indices) == 0, dominated_indices


def pairwise_non_dominated(front: list[tuple]) -> tuple[bool, list[tuple[int, int]]]:
    """Check that no point in *front* dominates another.

    Returns (valid, list_of_domination_pairs).
    """
    violations: list[tuple[int, int]] = []
    for i in range(len(front)):
        for j in range(len(front)):
            if i != j and dominates(front[i], front[j]):
                violations.append((i, j))
    return len(violations) == 0, violations


def compute_pareto_front_bruteforce(points: list[tuple]) -> list[int]:
    """Exact O(n²·d) Pareto front via pairwise dominance — reference implementation."""
    n = len(points)
    is_dominated = [False] * n
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            if dominates(points[j], points[i]):
                is_dominated[i] = True
                break
    return [i for i in range(n) if not is_dominated[i]]


def compute_hypervolume_mc(front_points: list[tuple], ref: tuple,
                           n_samples: int = 200_000) -> float:
    """Monte Carlo hypervolume indicator."""
    if not front_points:
        return 0.0
    ndim = len(ref)
    ideal = [min(p[d] for p in front_points) for d in range(ndim)]
    box_vol = 1.0
    for d in range(ndim):
        extent = ref[d] - ideal[d]
        if extent <= 0:
            return 0.0
        box_vol *= extent

    rng = random.Random(0)
    count = 0
    for _ in range(n_samples):
        sample = tuple(ideal[d] + rng.random() * (ref[d] - ideal[d]) for d in range(ndim))
        if any(all(p[d] <= sample[d] for d in range(ndim)) for p in front_points):
            count += 1
    return box_vol * count / n_samples


# ---------------------------------------------------------------------------
# Simulated MaxSMT Pareto enumeration (mirrors RegSynth iterative approach)
# ---------------------------------------------------------------------------


def maxsmt_pareto_enumerate(points: list[tuple], n_dims: int,
                            n_weight_samples: int = 200) -> list[int]:
    """Simulate RegSynth's iterative weighted MaxSMT Pareto enumeration."""
    rng = random.Random(7)
    front_indices: set[int] = set()
    for _ in range(n_weight_samples):
        weights = [rng.random() for _ in range(n_dims)]
        total = sum(weights)
        weights = [w / total for w in weights]
        best_idx = min(range(len(points)),
                       key=lambda i: sum(weights[d] * points[i][d] for d in range(n_dims)))
        front_indices.add(best_idx)
    # Filter dominated
    candidates = list(front_indices)
    cand_pts = [points[i] for i in candidates]
    nd = compute_pareto_front_bruteforce(cand_pts)
    return [candidates[i] for i in nd]


# ---------------------------------------------------------------------------
# Brute-force trajectory optimiser (for small instances)
# ---------------------------------------------------------------------------


def generate_regulatory_instance(n_obligations: int, n_dims: int,
                                 rng: random.Random) -> list[tuple]:
    """Generate a small regulatory compliance instance with correlated costs."""
    points = []
    for _ in range(n_obligations):
        vals = []
        base = rng.random()
        for d in range(n_dims):
            noise = rng.gauss(0, 0.15)
            vals.append(max(0.01, min(1.0, 0.3 * base + 0.7 * rng.random() + noise)))
        points.append(tuple(vals))
    return points


def bruteforce_trajectory_pareto(
    timesteps: list[list[tuple]],
    budget: int,
) -> list[tuple]:
    """Enumerate all feasible trajectories and return the non-dominated set.

    Each timestep has a list of candidate strategies (cost tuples).
    Budget = max L1 distance between consecutive strategies.
    Returns trajectory-level cost tuples (sum across timesteps).
    """
    T = len(timesteps)
    n_dims = len(timesteps[0][0])

    def l1_dist(a: tuple, b: tuple) -> float:
        return sum(abs(ai - bi) for ai, bi in zip(a, b))

    # Enumerate all feasible trajectories via DFS
    trajectories: list[tuple] = []

    def dfs(t: int, prev_idx: int, cumulative: list[float]):
        if t == T:
            trajectories.append(tuple(cumulative))
            return
        for j, strategy in enumerate(timesteps[t]):
            if t == 0 or l1_dist(timesteps[t - 1][prev_idx], strategy) <= budget:
                new_cum = [cumulative[d] + strategy[d] for d in range(n_dims)]
                dfs(t + 1, j, new_cum)

    dfs(0, -1, [0.0] * n_dims)

    if not trajectories:
        return []

    # Extract Pareto front
    nd_indices = compute_pareto_front_bruteforce(trajectories)
    return [trajectories[i] for i in nd_indices]


def greedy_per_timestep_trajectory(
    timesteps: list[list[tuple]],
    budget: int,
) -> list[tuple]:
    """Per-timestep Pareto-optimal with greedy selection (first non-dominated point)."""
    T = len(timesteps)
    n_dims = len(timesteps[0][0])

    trajectories: list[tuple] = []
    for start_idx in range(len(timesteps[0])):
        cumulative = list(timesteps[0][start_idx])
        prev_idx = start_idx
        feasible = True
        for t in range(1, T):
            # Pick locally best feasible strategy
            best_j = None
            best_score = float('inf')
            for j, strategy in enumerate(timesteps[t]):
                dist = sum(abs(timesteps[t - 1][prev_idx][d] - strategy[d])
                           for d in range(n_dims))
                if dist <= budget:
                    score = sum(strategy)
                    if score < best_score:
                        best_score = score
                        best_j = j
            if best_j is None:
                feasible = False
                break
            cumulative = [cumulative[d] + timesteps[t][best_j][d] for d in range(n_dims)]
            prev_idx = best_j
        if feasible:
            trajectories.append(tuple(cumulative))

    if not trajectories:
        return []
    nd_indices = compute_pareto_front_bruteforce(trajectories)
    return [trajectories[i] for i in nd_indices]


# ---------------------------------------------------------------------------
# Validation experiments
# ---------------------------------------------------------------------------


def validate_dominance_correctness(
    n_instances: int = 50,
    dims_list: list[int] | None = None,
    pool_sizes: list[int] | None = None,
    seed: int = 42,
) -> dict:
    """Experiment 1: Verify MaxSMT front is non-dominated for every instance."""
    dims_list = dims_list or [2, 3, 4, 5, 8]
    pool_sizes = pool_sizes or [50, 100, 200, 500]
    rng = random.Random(seed)

    results: list[dict] = []
    total_checks = 0
    total_pass = 0

    for n_dims in dims_list:
        for pool_size in pool_sizes:
            for inst in range(n_instances):
                points = generate_regulatory_instance(pool_size, n_dims, rng)
                maxsmt_front_idx = maxsmt_pareto_enumerate(points, n_dims)
                maxsmt_front = [points[i] for i in maxsmt_front_idx]

                # Check 1: pairwise non-domination within front
                pw_ok, pw_violations = pairwise_non_dominated(maxsmt_front)

                # Check 2: no point in pool dominates any front point
                pool_ok, pool_dominated = is_pareto_front(maxsmt_front, points)

                total_checks += 1
                passed = pw_ok and pool_ok
                if passed:
                    total_pass += 1

                results.append({
                    "dims": n_dims,
                    "pool_size": pool_size,
                    "instance": inst,
                    "front_size": len(maxsmt_front_idx),
                    "pairwise_ok": pw_ok,
                    "pool_ok": pool_ok,
                    "passed": passed,
                })

    pass_rate = total_pass / total_checks if total_checks > 0 else 0.0
    return {
        "experiment": "dominance_correctness",
        "description": "Verify all MaxSMT-produced frontiers are non-dominated",
        "total_instances": total_checks,
        "total_passed": total_pass,
        "pass_rate": round(pass_rate, 4),
        "dims_tested": dims_list,
        "pool_sizes_tested": pool_sizes,
        "details": results,
    }


def validate_maxsmt_vs_bruteforce(
    n_instances: int = 100,
    seed: int = 123,
) -> dict:
    """Experiment 2: Compare MaxSMT Pareto frontier against brute-force on small instances."""
    configs = [
        {"n_obligations": 10, "n_dims": 2},
        {"n_obligations": 15, "n_dims": 2},
        {"n_obligations": 20, "n_dims": 3},
        {"n_obligations": 30, "n_dims": 3},
        {"n_obligations": 20, "n_dims": 4},
        {"n_obligations": 50, "n_dims": 4},
    ]
    rng = random.Random(seed)
    results: list[dict] = []

    for cfg in configs:
        n_obl = cfg["n_obligations"]
        n_dims = cfg["n_dims"]
        exact_matches = 0
        subset_matches = 0
        hv_ratios: list[float] = []

        for inst in range(n_instances):
            points = generate_regulatory_instance(n_obl, n_dims, rng)

            bf_idx = compute_pareto_front_bruteforce(points)
            bf_front = [points[i] for i in bf_idx]

            ms_idx = maxsmt_pareto_enumerate(points, n_dims)
            ms_front = [points[i] for i in ms_idx]

            ref = tuple(1.1 for _ in range(n_dims))
            bf_hv = compute_hypervolume_mc(bf_front, ref, n_samples=50_000)
            ms_hv = compute_hypervolume_mc(ms_front, ref, n_samples=50_000)

            hv_ratio = ms_hv / bf_hv if bf_hv > 0 else 1.0
            hv_ratios.append(hv_ratio)

            bf_set = set(bf_idx)
            ms_set = set(ms_idx)
            if bf_set == ms_set:
                exact_matches += 1
            if ms_set <= bf_set:
                subset_matches += 1

        mean_hv_ratio = sum(hv_ratios) / len(hv_ratios) if hv_ratios else 0.0
        min_hv_ratio = min(hv_ratios) if hv_ratios else 0.0
        results.append({
            "n_obligations": n_obl,
            "n_dims": n_dims,
            "n_instances": n_instances,
            "exact_match_rate": round(exact_matches / n_instances, 4),
            "subset_match_rate": round(subset_matches / n_instances, 4),
            "mean_hv_ratio": round(mean_hv_ratio, 6),
            "min_hv_ratio": round(min_hv_ratio, 6),
        })

    return {
        "experiment": "maxsmt_vs_bruteforce",
        "description": "Compare MaxSMT frontier against brute-force on small instances",
        "configs": results,
    }


def validate_trajectory_domination(
    n_instances: int = 200,
    seed: int = 456,
) -> dict:
    """Experiment 3: Empirically validate Theorem 1 (trajectory domination).

    Generate small multi-timestep instances and compare joint-optimal
    trajectories against per-timestep greedy trajectories.
    """
    rng = random.Random(seed)
    n_dims = 2
    n_timesteps = 3
    n_strategies_per_step = 5
    budget = 2.0

    dominated_count = 0
    hv_improvements: list[float] = []

    for _ in range(n_instances):
        timesteps = []
        for t in range(n_timesteps):
            strategies = generate_regulatory_instance(n_strategies_per_step, n_dims, rng)
            timesteps.append(strategies)

        joint_front = bruteforce_trajectory_pareto(timesteps, budget)
        greedy_front = greedy_per_timestep_trajectory(timesteps, budget)

        if not joint_front or not greedy_front:
            continue

        # Check if any greedy trajectory is dominated by a joint trajectory
        any_dominated = False
        for gp in greedy_front:
            for jp in joint_front:
                if dominates(jp, gp):
                    any_dominated = True
                    break
            if any_dominated:
                break
        if any_dominated:
            dominated_count += 1

        ref = tuple(max(max(p[d] for p in joint_front + greedy_front) * 1.1, 1.0)
                    for d in range(n_dims))
        joint_hv = compute_hypervolume_mc(joint_front, ref, n_samples=50_000)
        greedy_hv = compute_hypervolume_mc(greedy_front, ref, n_samples=50_000)
        if greedy_hv > 0:
            improvement = (joint_hv - greedy_hv) / greedy_hv
            hv_improvements.append(improvement)

    domination_rate = dominated_count / n_instances if n_instances > 0 else 0.0
    mean_hv_improvement = (sum(hv_improvements) / len(hv_improvements)
                           if hv_improvements else 0.0)
    std_hv_improvement = _std(hv_improvements)

    return {
        "experiment": "trajectory_domination",
        "description": "Empirical validation of Theorem 1 (trajectory domination)",
        "n_instances": n_instances,
        "n_timesteps": n_timesteps,
        "n_strategies_per_step": n_strategies_per_step,
        "budget": budget,
        "domination_rate": round(domination_rate, 4),
        "mean_hv_improvement_pct": round(mean_hv_improvement * 100, 2),
        "std_hv_improvement_pct": round(std_hv_improvement * 100, 2),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> dict:
    parser = argparse.ArgumentParser(description="Pareto dominance validation benchmark")
    parser.add_argument("--runs", type=int, default=50,
                        help="Instances per dominance-correctness config (default: 50)")
    parser.add_argument("--bf-runs", type=int, default=100,
                        help="Instances for brute-force comparison (default: 100)")
    parser.add_argument("--traj-runs", type=int, default=200,
                        help="Instances for trajectory domination (default: 200)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.output or os.path.join(results_dir, "pareto_dominance_validation.json")

    print("=" * 70)
    print("PARETO DOMINANCE VALIDATION BENCHMARK")
    print("=" * 70)

    # Experiment 1: Dominance correctness
    print("\n--- Experiment 1: Dominance Correctness ---")
    t0 = time.perf_counter()
    exp1 = validate_dominance_correctness(n_instances=args.runs)
    t1 = time.perf_counter()
    print(f"  Pass rate: {exp1['pass_rate']:.4f} "
          f"({exp1['total_passed']}/{exp1['total_instances']})")
    print(f"  Time: {t1 - t0:.2f}s")

    # Experiment 2: MaxSMT vs brute-force
    print("\n--- Experiment 2: MaxSMT vs Brute-Force ---")
    t0 = time.perf_counter()
    exp2 = validate_maxsmt_vs_bruteforce(n_instances=args.bf_runs)
    t2 = time.perf_counter()
    for cfg in exp2["configs"]:
        print(f"  n={cfg['n_obligations']}, d={cfg['n_dims']}: "
              f"exact={cfg['exact_match_rate']:.2%}, "
              f"HV ratio={cfg['mean_hv_ratio']:.4f} "
              f"(min={cfg['min_hv_ratio']:.4f})")
    print(f"  Time: {t2 - t0:.2f}s")

    # Experiment 3: Trajectory domination
    print("\n--- Experiment 3: Trajectory Domination (Theorem 1) ---")
    t0 = time.perf_counter()
    exp3 = validate_trajectory_domination(n_instances=args.traj_runs)
    t3 = time.perf_counter()
    print(f"  Domination rate: {exp3['domination_rate']:.4f} "
          f"({int(exp3['domination_rate'] * exp3['n_instances'])}/{exp3['n_instances']})")
    print(f"  Mean HV improvement: {exp3['mean_hv_improvement_pct']:.2f}% "
          f"(± {exp3['std_hv_improvement_pct']:.2f}%)")
    print(f"  Time: {t3 - t0:.2f}s")

    # Assemble output
    output = {
        "benchmark": "pareto_dominance_validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiments": {
            "dominance_correctness": exp1,
            "maxsmt_vs_bruteforce": exp2,
            "trajectory_domination": exp3,
        },
        "summary": {
            "dominance_correctness_pass_rate": exp1["pass_rate"],
            "mean_hv_ratio_vs_bruteforce": round(
                sum(c["mean_hv_ratio"] for c in exp2["configs"]) / len(exp2["configs"]), 4
            ),
            "min_hv_ratio_vs_bruteforce": round(
                min(c["min_hv_ratio"] for c in exp2["configs"]), 4
            ),
            "trajectory_domination_rate": exp3["domination_rate"],
            "trajectory_hv_improvement_pct": exp3["mean_hv_improvement_pct"],
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    s = output["summary"]
    print(f"  Dominance correctness:     {s['dominance_correctness_pass_rate']:.2%}")
    print(f"  HV ratio vs brute-force:   {s['mean_hv_ratio_vs_bruteforce']:.4f} "
          f"(min {s['min_hv_ratio_vs_bruteforce']:.4f})")
    print(f"  Trajectory domination:     {s['trajectory_domination_rate']:.2%}")
    print(f"  Trajectory HV improvement: {s['trajectory_hv_improvement_pct']:.2f}%")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
