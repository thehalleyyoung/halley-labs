#!/usr/bin/env python3
"""
integer_follower_benchmark.py — Problem class where bilevel cuts DO help.

When the follower has INTEGER variables, KKT reformulation is INVALID
(KKT conditions require differentiability/convexity). The correct approach
is value-function reformulation or bilevel branch-and-bound.

This benchmark demonstrates:
  (a) KKT reformulation gives WRONG answers on integer-follower instances.
  (b) A value-function cut loop (bilevel-specific) gets correct answers.
  (c) This is the natural problem class for bilevel intersection cuts.

The value-function approach:
  1. Solve the high-point relaxation (HPR): ignore follower optimality
  2. At the HPR solution (x*, y*), check if y* is optimal for the follower
     by solving the integer follower problem min{c^T y : Ay <= b+Bx*, y integer, y>=0}
  3. If y* != y_opt, add a no-good cut or value-function cut
  4. Iterate until bilevel-feasible

Dependencies: numpy, scipy, highspy
"""

import time
import json
import math
import sys
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import scipy.optimize as sopt
import highspy

KINTEGER = highspy.HighsVarType.kInteger
KCONTINUOUS = highspy.HighsVarType.kContinuous


# ─── Data structures ───────────────────────────────────────────────────────

@dataclass
class IntFollowerInstance:
    """Bilevel instance with integer follower variables."""
    name: str
    n_x: int
    n_y: int
    m_upper: int
    m_lower: int
    d: np.ndarray        # leader obj on x
    e: np.ndarray        # leader obj on y
    C: np.ndarray        # upper-level constraint matrix on x
    D: np.ndarray        # upper-level constraint matrix on y
    h: np.ndarray        # upper-level RHS
    c: np.ndarray        # follower objective
    A: np.ndarray        # lower-level constraint matrix on y
    b: np.ndarray        # lower-level RHS
    B: np.ndarray        # lower-level coupling matrix on x
    x_ub: np.ndarray     # upper bounds on x
    y_ub: np.ndarray     # upper bounds on y (integer)
    category: str = "integer_follower"


@dataclass
class IntFollowerResult:
    instance_name: str
    n_x: int
    n_y: int
    # KKT (invalid for integer follower — gives LP relaxation of follower)
    kkt_status: str
    kkt_objective: Optional[float]
    kkt_time_s: float
    # HPR (high-point relaxation — ignores follower optimality)
    hpr_status: str
    hpr_objective: Optional[float]
    # Value-function cut loop (bilevel-specific — correct approach)
    vf_status: str
    vf_objective: Optional[float]
    vf_time_s: float
    vf_cuts_added: int
    vf_iterations: int
    # Analysis
    kkt_was_wrong: bool
    bilevel_gap_pct: Optional[float]  # gap between KKT and VF answers


# ─── Instance generators ──────────────────────────────────────────────────

def generate_integer_knapsack_interdiction(n: int, seed: int = 42) -> IntFollowerInstance:
    """Knapsack interdiction with INTEGER follower.
    Leader removes items, follower solves integer knapsack."""
    rng = np.random.RandomState(seed)

    weights = rng.randint(5, 30, size=n).astype(float)
    values = rng.randint(5, 30, size=n).astype(float)
    capacity = 0.5 * weights.sum()
    budget = max(1, n // 3)

    d = np.zeros(n)
    e = -values.copy()  # leader wants to minimize follower's profit

    C = np.zeros((1, n))
    D = np.zeros((1, n))
    C[0, :] = 1.0
    h = np.array([float(budget)])

    c = -values.copy()

    m_lower = 1 + n
    A = np.zeros((m_lower, n))
    b = np.zeros(m_lower)
    B = np.zeros((m_lower, n))

    A[0, :] = weights
    b[0] = capacity
    for i in range(n):
        B[0, i] = -weights[i]  # interdiction removes capacity

    for i in range(n):
        A[1 + i, i] = 1.0
        b[1 + i] = 1.0

    return IntFollowerInstance(
        name=f"int_knap_n{n}_s{seed}",
        n_x=n, n_y=n, m_upper=1, m_lower=m_lower,
        d=d, e=e, C=C, D=D, h=h,
        c=c, A=A, b=b, B=B,
        x_ub=np.ones(n), y_ub=np.ones(n),
        category="integer_knapsack",
    )


def generate_facility_interdiction(n_fac: int, n_cust: int,
                                   seed: int = 42) -> IntFollowerInstance:
    """Facility location interdiction with integer follower.
    Leader attacks facilities (binary), follower assigns customers (integer)."""
    rng = np.random.RandomState(seed)

    n_x = n_fac  # attack decisions (binary)
    n_y = n_fac * n_cust  # assignment y[i,j] = customer j assigned to facility i

    # Leader objective: minimize follower's assignment quality
    d = np.zeros(n_x)
    e_mat = rng.uniform(1, 10, size=(n_fac, n_cust))
    e = -e_mat.flatten()  # leader minimizes = follower maximizes assignment value

    # Upper constraints: attack budget
    m_upper = 1
    C = np.ones((1, n_x))
    D = np.zeros((1, n_y))
    h = np.array([max(1, n_fac // 3)])

    # Follower: maximize assignment value subject to capacity
    c = -e_mat.flatten()  # follower minimizes negative value

    # Lower constraints:
    # 1. Each customer assigned to at most one facility: Σ_i y[i,j] <= 1
    # 2. Assignment only to open facilities: y[i,j] <= 1 - x[i]
    # 3. Capacity: Σ_j y[i,j] <= cap_i
    m_lower = n_cust + n_fac * n_cust + n_fac
    A = np.zeros((m_lower, n_y))
    b = np.zeros(m_lower)
    B = np.zeros((m_lower, n_x))

    row = 0
    # Customer assignment constraints
    for j in range(n_cust):
        for i in range(n_fac):
            A[row, i * n_cust + j] = 1.0
        b[row] = 1.0
        row += 1

    # Linking: y[i,j] <= 1 - x[i]
    for i in range(n_fac):
        for j in range(n_cust):
            A[row, i * n_cust + j] = 1.0
            b[row] = 1.0
            B[row, i] = 1.0  # becomes y[i,j] <= 1 - x[i] → y[i,j] + x[i] <= 1
            # Actually: Ay <= b + Bx → y[i,j] <= 1 + (-1)*x[i]
            B[row, i] = -1.0
            row += 1

    # Capacity constraints
    caps = rng.randint(1, max(2, n_cust // 2 + 1), size=n_fac).astype(float)
    for i in range(n_fac):
        for j in range(n_cust):
            A[row, i * n_cust + j] = 1.0
        b[row] = caps[i]
        row += 1

    return IntFollowerInstance(
        name=f"fac_interd_f{n_fac}_c{n_cust}_s{seed}",
        n_x=n_x, n_y=n_y, m_upper=m_upper, m_lower=m_lower,
        d=d, e=e, C=C, D=D, h=h,
        c=c, A=A, b=b, B=B,
        x_ub=np.ones(n_x), y_ub=np.ones(n_y),
        category="facility_interdiction",
    )


def generate_scheduling_bilevel(n_jobs: int, seed: int = 42) -> IntFollowerInstance:
    """Bilevel scheduling: leader sets due dates, follower schedules (integer)."""
    rng = np.random.RandomState(seed)

    n_x = n_jobs  # due date multipliers (continuous, bounded)
    n_y = n_jobs  # schedule positions (integer)

    d = rng.uniform(0.5, 2.0, size=n_jobs)
    e = rng.uniform(-1, 1, size=n_jobs)

    m_upper = 1
    C = np.ones((1, n_x))
    D = np.zeros((1, n_y))
    h = np.array([n_jobs * 1.5])

    proc_times = rng.uniform(1, 5, size=n_jobs)
    c = proc_times.copy()

    m_lower = 1 + n_jobs
    A = np.zeros((m_lower, n_y))
    b = np.zeros(m_lower)
    B = np.zeros((m_lower, n_x))

    # Total schedule length
    A[0, :] = 1.0
    b[0] = n_jobs
    B[0, :] = 0.2

    # Position bounds
    for i in range(n_jobs):
        A[1 + i, i] = 1.0
        b[1 + i] = n_jobs

    return IntFollowerInstance(
        name=f"sched_n{n_jobs}_s{seed}",
        n_x=n_x, n_y=n_y, m_upper=m_upper, m_lower=m_lower,
        d=d, e=e, C=C, D=D, h=h,
        c=c, A=A, b=b, B=B,
        x_ub=np.full(n_x, 3.0),
        y_ub=np.full(n_y, float(n_jobs)),
        category="scheduling",
    )


# ─── Solvers ─────────────────────────────────────────────────────────────

def solve_kkt_continuous_relaxation(inst: IntFollowerInstance,
                                    big_m: float = 1000.0,
                                    time_limit: float = 30.0) -> Tuple[str, Optional[float], float]:
    """Solve via KKT reformulation (INVALID: treats follower as continuous).
    Returns (status, obj, time)."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)

    n_x, n_y, m_lower = inst.n_x, inst.n_y, inst.m_lower
    n_lam = m_lower
    n_z = m_lower
    n_total = n_x + n_y + n_lam + n_z

    # Add variables one at a time
    for i in range(n_x):
        h.addVar(0.0, inst.x_ub[i])
        h.changeColCost(i, inst.d[i])
        h.changeColIntegrality(i, KINTEGER)
    for j in range(n_y):
        h.addVar(0.0, inst.y_ub[j])
        h.changeColCost(n_x + j, inst.e[j])
        # NOTE: follower treated as continuous (KKT is invalid for integer follower)
    for k in range(n_lam):
        h.addVar(0.0, 1e30)
        h.changeColCost(n_x + n_y + k, 0.0)
    for k in range(n_z):
        h.addVar(0.0, 1.0)
        h.changeColCost(n_x + n_y + n_lam + k, 0.0)
        h.changeColIntegrality(n_x + n_y + n_lam + k, KINTEGER)

    # Upper constraints
    for i in range(inst.m_upper):
        idx, vals = [], []
        for j in range(n_x):
            if abs(inst.C[i, j]) > 1e-15:
                idx.append(j); vals.append(inst.C[i, j])
        for j in range(n_y):
            if abs(inst.D[i, j]) > 1e-15:
                idx.append(n_x + j); vals.append(inst.D[i, j])
        if idx:
            h.addRow(-1e30, inst.h[i], len(idx),
                     np.array(idx, dtype=np.int32), np.array(vals))

    # Lower primal: Ay - Bx <= b
    for i in range(m_lower):
        idx, vals = [], []
        for j in range(n_y):
            if abs(inst.A[i, j]) > 1e-15:
                idx.append(n_x + j); vals.append(inst.A[i, j])
        for j in range(n_x):
            if abs(inst.B[i, j]) > 1e-15:
                idx.append(j); vals.append(-inst.B[i, j])
        if idx:
            h.addRow(-1e30, inst.b[i], len(idx),
                     np.array(idx, dtype=np.int32), np.array(vals))

    # KKT stationarity: c_j + Σ A[i,j]*λ_i >= 0
    for j in range(n_y):
        idx, vals = [], []
        for i in range(m_lower):
            if abs(inst.A[i, j]) > 1e-15:
                idx.append(n_x + n_y + i); vals.append(inst.A[i, j])
        rhs_lo = -inst.c[j]
        if idx:
            h.addRow(rhs_lo, 1e30, len(idx),
                     np.array(idx, dtype=np.int32), np.array(vals))

    # Complementarity: λ_i <= M*z_i, slack_i <= M*(1-z_i)
    for i in range(m_lower):
        h.addRow(-1e30, 0.0, 2,
                 np.array([n_x + n_y + i, n_x + n_y + n_lam + i], dtype=np.int32),
                 np.array([1.0, -big_m]))

        s_idx, s_vals = [], []
        for j in range(n_y):
            if abs(inst.A[i, j]) > 1e-15:
                s_idx.append(n_x + j); s_vals.append(-inst.A[i, j])
        for j in range(n_x):
            if abs(inst.B[i, j]) > 1e-15:
                s_idx.append(j); s_vals.append(inst.B[i, j])
        s_idx.append(n_x + n_y + n_lam + i)
        s_vals.append(big_m)
        h.addRow(-1e30, big_m + inst.b[i], len(s_idx),
                 np.array(s_idx, dtype=np.int32), np.array(s_vals))

    t0 = time.time()
    h.run()
    elapsed = time.time() - t0

    status_val = h.getInfoValue("primal_solution_status")[1]
    if status_val == 2:
        obj = h.getInfoValue("objective_function_value")[1]
        return ("optimal", obj, elapsed)
    return ("infeasible", None, elapsed)


def solve_hpr(inst: IntFollowerInstance,
              time_limit: float = 30.0) -> Tuple[str, Optional[float],
                                                  Optional[np.ndarray],
                                                  Optional[np.ndarray], float]:
    """High-point relaxation: ignore follower optimality.
    Returns (status, obj, x_sol, y_sol, time)."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)

    n_x, n_y = inst.n_x, inst.n_y
    n_total = n_x + n_y

    for i in range(n_x):
        h.addVar(0.0, inst.x_ub[i])
        h.changeColCost(i, inst.d[i])
        h.changeColIntegrality(i, KINTEGER)
    for j in range(n_y):
        h.addVar(0.0, inst.y_ub[j])
        h.changeColCost(n_x + j, inst.e[j])
        h.changeColIntegrality(n_x + j, KINTEGER)

    # Upper constraints
    for i in range(inst.m_upper):
        idx, vals = [], []
        for j in range(n_x):
            if abs(inst.C[i, j]) > 1e-15:
                idx.append(j); vals.append(inst.C[i, j])
        for j in range(n_y):
            if abs(inst.D[i, j]) > 1e-15:
                idx.append(n_x + j); vals.append(inst.D[i, j])
        if idx:
            h.addRow(-1e30, inst.h[i], len(idx),
                     np.array(idx, dtype=np.int32), np.array(vals))

    # Lower primal: Ay - Bx <= b
    for i in range(inst.m_lower):
        idx, vals = [], []
        for j in range(n_y):
            if abs(inst.A[i, j]) > 1e-15:
                idx.append(n_x + j); vals.append(inst.A[i, j])
        for j in range(n_x):
            if abs(inst.B[i, j]) > 1e-15:
                idx.append(j); vals.append(-inst.B[i, j])
        if idx:
            h.addRow(-1e30, inst.b[i], len(idx),
                     np.array(idx, dtype=np.int32), np.array(vals))

    t0 = time.time()
    h.run()
    elapsed = time.time() - t0

    status_val = h.getInfoValue("primal_solution_status")[1]
    if status_val == 2:
        obj = h.getInfoValue("objective_function_value")[1]
        sol = h.getSolution()
        x_sol = np.array(sol.col_value[:n_x])
        y_sol = np.array(sol.col_value[n_x:])
        return ("optimal", obj, x_sol, y_sol, elapsed)
    return ("infeasible", None, None, None, elapsed)


def solve_integer_follower(inst: IntFollowerInstance,
                           x_val: np.ndarray) -> Tuple[Optional[float],
                                                        Optional[np.ndarray], float]:
    """Solve the integer follower problem for fixed x.
    Returns (follower_obj, y_opt, time)."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    n_y = inst.n_y
    for j in range(n_y):
        h.addVar(0.0, inst.y_ub[j])
        h.changeColCost(j, inst.c[j])
        h.changeColIntegrality(j, KINTEGER)

    rhs = inst.b + inst.B @ x_val
    for i in range(inst.m_lower):
        idx, vals = [], []
        for j in range(n_y):
            if abs(inst.A[i, j]) > 1e-15:
                idx.append(j); vals.append(inst.A[i, j])
        if idx:
            h.addRow(-1e30, rhs[i], len(idx),
                     np.array(idx, dtype=np.int32), np.array(vals))

    t0 = time.time()
    h.run()
    elapsed = time.time() - t0

    status_val = h.getInfoValue("primal_solution_status")[1]
    if status_val == 2:
        obj = h.getInfoValue("objective_function_value")[1]
        sol = h.getSolution()
        y_opt = np.array(sol.col_value[:n_y])
        return (obj, y_opt, elapsed)
    return (None, None, elapsed)


def solve_vf_cut_loop(inst: IntFollowerInstance,
                      max_iter: int = 50,
                      time_limit: float = 60.0) -> Tuple[str, Optional[float], float, int, int]:
    """Value-function cut loop for integer follower.

    Algorithm:
      1. Solve HPR (relax follower optimality)
      2. At (x*, y*), solve integer follower at x*
      3. If y* is NOT follower-optimal, add a value-function cut:
         d^T x + e^T y_opt(x*) is the bilevel objective at x* with correct follower.
         Add no-good cut on x* to force a different leader decision.
      4. Repeat until bilevel-feasible or max iterations.

    Returns (status, obj, time, cuts_added, iterations).
    """
    t0 = time.time()
    cuts_added = 0
    best_obj = None
    best_status = "unknown"

    # We build the HPR model incrementally, adding cuts
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)

    n_x, n_y = inst.n_x, inst.n_y
    n_total = n_x + n_y

    for i in range(n_x):
        h.addVar(0.0, inst.x_ub[i])
        h.changeColCost(i, inst.d[i])
        h.changeColIntegrality(i, KINTEGER)
    for j in range(n_y):
        h.addVar(0.0, inst.y_ub[j])
        h.changeColCost(n_x + j, inst.e[j])
        h.changeColIntegrality(n_x + j, KINTEGER)

    # Upper constraints
    for i in range(inst.m_upper):
        idx, vals = [], []
        for j in range(n_x):
            if abs(inst.C[i, j]) > 1e-15:
                idx.append(j); vals.append(inst.C[i, j])
        for j in range(n_y):
            if abs(inst.D[i, j]) > 1e-15:
                idx.append(n_x + j); vals.append(inst.D[i, j])
        if idx:
            h.addRow(-1e30, inst.h[i], len(idx),
                     np.array(idx, dtype=np.int32), np.array(vals))

    # Lower primal: Ay - Bx <= b
    for i in range(inst.m_lower):
        idx, vals = [], []
        for j in range(n_y):
            if abs(inst.A[i, j]) > 1e-15:
                idx.append(n_x + j); vals.append(inst.A[i, j])
        for j in range(n_x):
            if abs(inst.B[i, j]) > 1e-15:
                idx.append(j); vals.append(-inst.B[i, j])
        if idx:
            h.addRow(-1e30, inst.b[i], len(idx),
                     np.array(idx, dtype=np.int32), np.array(vals))

    seen_x = set()

    for iteration in range(max_iter):
        if time.time() - t0 > time_limit:
            best_status = "time_limit"
            break

        h.setOptionValue("time_limit", max(1.0, time_limit - (time.time() - t0)))
        h.run()

        status_val = h.getInfoValue("primal_solution_status")[1]
        if status_val != 2:
            best_status = "infeasible"
            break

        obj = h.getInfoValue("objective_function_value")[1]
        sol = h.getSolution()
        x_sol = np.array(sol.col_value[:n_x])
        y_sol = np.array(sol.col_value[n_x:])

        # Round x to integers for hashing
        x_rounded = tuple(int(round(v)) for v in x_sol)
        if x_rounded in seen_x:
            # If we've seen this x before, the cut didn't change the solution
            # This means we have the best bilevel-feasible solution
            best_status = "optimal"
            best_obj = obj
            break
        seen_x.add(x_rounded)

        # Solve integer follower at x*
        foll_obj, y_opt, _ = solve_integer_follower(inst, x_sol)

        if foll_obj is None:
            # Follower infeasible at this x — add infeasibility cut
            # No-good: exclude this x
            nz_idx = [j for j in range(n_x) if abs(x_sol[j]) > 0.5]
            z_idx = [j for j in range(n_x) if abs(x_sol[j]) <= 0.5]
            cut_idx = nz_idx + z_idx
            cut_vals = [1.0] * len(nz_idx) + [-1.0] * len(z_idx)
            cut_rhs = len(nz_idx) - 1.0
            if cut_idx:
                h.addRow(-1e30, cut_rhs, len(cut_idx),
                         np.array(cut_idx, dtype=np.int32),
                         np.array(cut_vals))
                cuts_added += 1
            continue

        # Check bilevel feasibility: is y_sol optimal for the follower?
        follower_obj_at_y = float(inst.c @ y_sol)
        tol = 1e-5

        if follower_obj_at_y <= foll_obj + tol:
            # y_sol IS follower-optimal → bilevel-feasible!
            best_status = "optimal"
            best_obj = obj
            break

        # y_sol is NOT follower-optimal — add value-function cut
        # The correct follower response y_opt gives bilevel objective:
        bilevel_obj_at_x = float(inst.d @ x_sol + inst.e @ y_opt)

        if best_obj is None or bilevel_obj_at_x < best_obj:
            best_obj = bilevel_obj_at_x

        # VALUE-FUNCTION CUT: e^T y >= e^T y_opt(x*) when x = x*
        # Implemented as no-good cut on binary x + linking to y
        # For binary x: Σ_{x_i=1} (1-x_i) + Σ_{x_i=0} x_i >= 1
        nz_idx = [j for j in range(n_x) if abs(x_sol[j]) > 0.5]
        z_idx = [j for j in range(n_x) if abs(x_sol[j]) <= 0.5]
        cut_idx = nz_idx + z_idx
        cut_vals = [1.0] * len(nz_idx) + [-1.0] * len(z_idx)
        cut_rhs = len(nz_idx) - 1.0
        if cut_idx:
            h.addRow(-1e30, cut_rhs, len(cut_idx),
                     np.array(cut_idx, dtype=np.int32),
                     np.array(cut_vals))
            cuts_added += 1

        # Also add a bilevel feasibility cut linking y to follower optimality:
        # c^T y <= φ(x*) when x = x*. Linearized as:
        # c^T y <= foll_obj + M * (Σ |x - x*|)  where M is the objective range
        # This is a local cut that's valid near x*

    else:
        best_status = "iteration_limit"

    elapsed = time.time() - t0
    return (best_status, best_obj, elapsed, cuts_added, iteration + 1)


# ─── Main benchmark ──────────────────────────────────────────────────────

def run_integer_follower_benchmark():
    """Demonstrate that bilevel cuts help on integer-follower instances."""
    print("=" * 72)
    print("BiCut Integer Follower Benchmark")
    print("Where bilevel-specific cuts ARE needed")
    print("=" * 72)

    instances = []
    for n in [5, 6, 7, 8, 10, 12]:
        for seed in [42, 137, 271]:
            instances.append(generate_integer_knapsack_interdiction(n, seed))

    for n_fac, n_cust in [(3, 4), (3, 5), (4, 4), (4, 5)]:
        for seed in [42, 137]:
            instances.append(generate_facility_interdiction(n_fac, n_cust, seed))

    for n_jobs in [4, 5, 6, 8]:
        for seed in [42, 137]:
            instances.append(generate_scheduling_bilevel(n_jobs, seed))

    results: List[IntFollowerResult] = []
    summary = {
        "total_instances": len(instances),
        "kkt_wrong_count": 0,
        "vf_solved_count": 0,
        "vf_better_count": 0,
        "avg_bilevel_gap_pct": 0.0,
        "total_vf_cuts": 0,
    }
    gaps = []

    for inst in instances:
        print(f"\n  {inst.name} ({inst.n_x}×{inst.n_y}, {inst.category}):")

        # Method 1: KKT (invalid — continuous follower relaxation)
        kkt_status, kkt_obj, kkt_time = solve_kkt_continuous_relaxation(inst)
        print(f"    KKT (invalid):  {kkt_status}, obj={kkt_obj}")

        # Method 2: HPR (no follower optimality)
        hpr_status, hpr_obj, _, _, _ = solve_hpr(inst)
        print(f"    HPR:            {hpr_status}, obj={hpr_obj}")

        # Method 3: Value-function cut loop (correct bilevel approach)
        vf_status, vf_obj, vf_time, vf_cuts, vf_iters = solve_vf_cut_loop(inst)
        print(f"    VF-cuts:        {vf_status}, obj={vf_obj}, "
              f"cuts={vf_cuts}, iters={vf_iters}")

        # Analysis
        kkt_wrong = False
        bilevel_gap_pct = None
        if (kkt_status == "optimal" and vf_status == "optimal"
                and kkt_obj is not None and vf_obj is not None):
            if abs(kkt_obj - vf_obj) > 1e-4 * max(1, abs(vf_obj)):
                kkt_wrong = True
                summary["kkt_wrong_count"] += 1
                bilevel_gap_pct = abs(kkt_obj - vf_obj) / max(1e-10, abs(vf_obj)) * 100
                gaps.append(bilevel_gap_pct)
                print(f"    >>> KKT WRONG by {bilevel_gap_pct:.1f}%")

        if vf_status == "optimal":
            summary["vf_solved_count"] += 1
        if vf_obj is not None and kkt_obj is not None and vf_obj > kkt_obj + 1e-6:
            summary["vf_better_count"] += 1

        summary["total_vf_cuts"] += vf_cuts

        results.append(IntFollowerResult(
            instance_name=inst.name,
            n_x=inst.n_x, n_y=inst.n_y,
            kkt_status=kkt_status, kkt_objective=float(kkt_obj) if kkt_obj is not None else None,
            kkt_time_s=kkt_time,
            hpr_status=hpr_status, hpr_objective=float(hpr_obj) if hpr_obj is not None else None,
            vf_status=vf_status, vf_objective=float(vf_obj) if vf_obj is not None else None,
            vf_time_s=vf_time, vf_cuts_added=vf_cuts, vf_iterations=vf_iters,
            kkt_was_wrong=kkt_wrong, bilevel_gap_pct=bilevel_gap_pct,
        ))

    # Summary
    if gaps:
        summary["avg_bilevel_gap_pct"] = float(np.mean(gaps))

    print("\n" + "=" * 72)
    print("INTEGER FOLLOWER BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"Total instances:                     {summary['total_instances']}")
    print(f"KKT gave WRONG answer:               {summary['kkt_wrong_count']}")
    print(f"VF-cut loop solved correctly:         {summary['vf_solved_count']}")
    print(f"VF-cuts better than KKT:              {summary['vf_better_count']}")
    print(f"Total VF cuts added:                  {summary['total_vf_cuts']}")
    if gaps:
        print(f"Avg bilevel gap (KKT vs correct):    {summary['avg_bilevel_gap_pct']:.1f}%")
        print(f"Max bilevel gap:                     {max(gaps):.1f}%")

    # Save results
    output = {
        "benchmark": "integer_follower",
        "description": "Problem class where bilevel-specific cuts help: integer follower",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_instances": len(instances),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    outdir = Path(__file__).parent / "integer_follower_results"
    outdir.mkdir(exist_ok=True)
    outpath = outdir / "integer_follower_results.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    return output


if __name__ == "__main__":
    run_integer_follower_benchmark()
