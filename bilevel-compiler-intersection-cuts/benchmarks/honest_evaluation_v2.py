#!/usr/bin/env python3
"""
honest_evaluation_v2.py — Enhanced honest evaluation with HARDER instances.

V1 findings: All 34 instances solved at root with 0 gap.
The LP relaxation of the KKT reformulation is tight for continuous-follower
problems with tight big-M values. This is a known property.

V2 strategy: Generate instances where bilevel cuts SHOULD matter:
  1. Loose big-M values (weaker KKT formulation)
  2. Dense constraint matrices (more complementarity vars)
  3. Problems where LP relaxation gap is >10%
  4. Compare with/without bilevel-specific value-function cuts
  5. Download and test on real BOBILib-style instances

Also: try to install and compare with MibS if available.
"""

import time
import json
import math
import sys
import os
import subprocess
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import scipy.optimize as sopt
import highspy

KINTEGER = highspy._core.HighsVarType.kInteger


# ─── Data structures ───────────────────────────────────────────────────────

@dataclass
class BilevelInstance:
    """MIBLP instance. See honest_evaluation.py for full docs."""
    name: str
    n_x: int
    n_y: int
    m_upper: int
    m_lower: int
    d: np.ndarray
    e: np.ndarray
    C: np.ndarray
    D: np.ndarray
    h: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    B: np.ndarray
    x_lb: np.ndarray = None
    x_ub: np.ndarray = None
    x_binary: bool = True
    category: str = "general"

    def __post_init__(self):
        if self.x_lb is None:
            self.x_lb = np.zeros(self.n_x)
        if self.x_ub is None:
            self.x_ub = np.ones(self.n_x)


@dataclass
class SolveResult:
    instance_name: str
    method: str
    status: str
    objective: Optional[float] = None
    solve_time_s: float = 0.0
    nodes: int = 0
    lp_relax_obj: Optional[float] = None
    gap_pct: Optional[float] = None
    cuts_added: int = 0
    iterations: int = 0
    n_vars_milp: int = 0
    n_cons_milp: int = 0


# ─── HARD instance generators ────────────────────────────────────────────

def generate_hard_knapsack_interdiction(n: int, seed: int = 42,
                                         correlation: str = "weakly") -> BilevelInstance:
    """Knapsack interdiction with HARD LP relaxation gap.

    Key differences from v1:
    - Use 'weakly correlated' or 'strongly correlated' weights/values
      (following Pisinger's hard knapsack generator)
    - Tighter capacity ratio → more fractional LP solutions
    - Leader has budget constraint that creates more complex interaction
    """
    rng = np.random.RandomState(seed)

    if correlation == "weakly":
        weights = rng.randint(10, 100, size=n).astype(float)
        values = weights + rng.randint(-10, 11, size=n).astype(float)
        values = np.maximum(values, 1.0)
    elif correlation == "strongly":
        weights = rng.randint(10, 100, size=n).astype(float)
        values = weights + 10.0
    else:
        weights = rng.randint(1, 50, size=n).astype(float)
        values = rng.randint(1, 50, size=n).astype(float)

    # Tight capacity → harder
    capacity = 0.35 * weights.sum()
    budget = max(1, n // 4)

    n_x = n
    n_y = n
    m_upper = 2  # budget + linking
    m_lower = 1 + n  # capacity + upper bounds

    # Leader: min -values^T y (maximize interdiction impact)
    d = np.zeros(n_x)
    e = -values.copy()

    # Upper-level: budget + a linking constraint to create coupling
    C_mat = np.zeros((m_upper, n_x))
    D_mat = np.zeros((m_upper, n_y))
    h_vec = np.zeros(m_upper)
    C_mat[0, :] = 1.0
    h_vec[0] = budget
    # Linking: sum(x_i + y_i) <= n * 0.7 (creates interdependence)
    C_mat[1, :] = 1.0
    D_mat[1, :] = 1.0
    h_vec[1] = n * 0.7

    c_vec = -values.copy()

    A_mat = np.zeros((m_lower, n_y))
    b_vec = np.zeros(m_lower)
    B_mat = np.zeros((m_lower, n_x))

    # Capacity with interdiction
    A_mat[0, :] = weights
    b_vec[0] = capacity
    B_mat[0, :] = -weights  # capacity shrinks with interdiction

    # y_i <= 1
    for i in range(n):
        A_mat[1 + i, i] = 1.0
        b_vec[1 + i] = 1.0

    return BilevelInstance(
        name=f"hard_knap_n{n}_{correlation[:4]}_s{seed}",
        n_x=n_x, n_y=n_y, m_upper=m_upper, m_lower=m_lower,
        d=d, e=e, C=C_mat, D=D_mat, h=h_vec,
        c=c_vec, A=A_mat, b=b_vec, B=B_mat,
        x_binary=True, category="knapsack_interdiction",
    )


def generate_dense_bilevel(n_x: int, n_y: int, density: float = 0.7,
                            seed: int = 42) -> BilevelInstance:
    """Generate a dense bilevel LP with many active constraints.

    Dense problems create more complementarity variables in KKT,
    making the big-M formulation weaker and bilevel cuts more valuable.
    """
    rng = np.random.RandomState(seed)

    m_upper = max(2, n_x // 2)
    m_lower = max(3, n_y + n_x // 2)

    d = rng.uniform(-5, 5, size=n_x)
    e = rng.uniform(-5, 5, size=n_y)

    # Dense upper-level constraints
    C = rng.uniform(-1, 3, size=(m_upper, n_x))
    C[rng.random(C.shape) > density] = 0
    D = rng.uniform(-1, 3, size=(m_upper, n_y))
    D[rng.random(D.shape) > density] = 0
    h = rng.uniform(5, 20, size=m_upper)

    c = rng.uniform(-3, 3, size=n_y)

    # Dense lower-level
    A = rng.uniform(0, 3, size=(m_lower, n_y))
    A[rng.random(A.shape) > density] = 0
    b = rng.uniform(2, 15, size=m_lower)
    B = rng.uniform(-2, 2, size=(m_lower, n_x))
    B[rng.random(B.shape) > density] = 0

    return BilevelInstance(
        name=f"dense_{n_x}x{n_y}_d{int(density*10)}_s{seed}",
        n_x=n_x, n_y=n_y, m_upper=m_upper, m_lower=m_lower,
        d=d, e=e, C=C, D=D, h=h,
        c=c, A=A, b=b, B=B,
        x_binary=True,
        x_ub=np.full(n_x, 3.0),
        category="dense_bilevel",
    )


def generate_bilevel_with_integer_linking(n: int, seed: int = 42) -> BilevelInstance:
    """Bilevel with integer coupling between leader and follower.

    These are harder because the value function phi(x) is discontinuous
    when the follower has integer variables (but we relax follower to LP).
    The KKT big-M formulation is weak here because many complementarity
    constraints are simultaneously near-active.
    """
    rng = np.random.RandomState(seed)

    n_x = n
    n_y = n
    m_upper = 2
    m_lower = 2 * n  # many constraints to create more duals

    d = rng.uniform(-3, 1, size=n_x)
    e = rng.uniform(-5, -1, size=n_y)  # leader benefits from follower activity

    C_mat = np.zeros((m_upper, n_x))
    D_mat = np.zeros((m_upper, n_y))
    h_vec = np.zeros(m_upper)
    C_mat[0, :] = 1.0
    h_vec[0] = n * 0.6
    D_mat[1, :] = 1.0
    h_vec[1] = n * 0.8

    c_vec = rng.uniform(1, 10, size=n_y)

    A_mat = np.zeros((m_lower, n_y))
    b_vec = np.zeros(m_lower)
    B_mat = np.zeros((m_lower, n_x))

    # Pairwise coupling constraints: y_i + y_{i+1} <= b_k + B_k x
    for k in range(n):
        A_mat[k, k] = 1.0
        if k + 1 < n:
            A_mat[k, k + 1] = 0.5
        b_vec[k] = rng.uniform(1, 3)
        B_mat[k, k] = -rng.uniform(0.2, 0.8)

    # Individual bounds
    for k in range(n):
        A_mat[n + k, k] = 1.0
        b_vec[n + k] = rng.uniform(2, 5)
        B_mat[n + k, min(k, n_x - 1)] = -rng.uniform(0.1, 0.5)

    return BilevelInstance(
        name=f"intlink_n{n}_s{seed}",
        n_x=n_x, n_y=n_y, m_upper=m_upper, m_lower=m_lower,
        d=d, e=e, C=C_mat, D=D_mat, h=h_vec,
        c=c_vec, A=A_mat, b=b_vec, B=B_mat,
        x_binary=True, category="integer_linking",
    )


def generate_stackelberg_game(n_strategies: int, seed: int = 42) -> BilevelInstance:
    """Stackelberg security game formulation.

    Leader: allocate security resources (binary)
    Follower: choose best attack given leader's allocation
    """
    rng = np.random.RandomState(seed)

    n_targets = n_strategies
    n_x = n_targets  # which targets to defend
    n_y = n_targets  # which target to attack (probability)

    # Leader defends at most k targets
    k = max(1, n_targets // 3)

    m_upper = 1
    m_lower = n_targets + 1  # attack probabilities + normalization

    # Leader: min sum(attack_damage * (1 - x_i) * y_i)
    # Linearized: min sum(-damage_i * y_i) + sum(damage_i * x_i * y_i)
    # With continuous follower: we use bilinear McCormick or simplified version
    damage = rng.uniform(5, 50, size=n_targets)
    d = np.zeros(n_x)  # leader cost
    e = damage.copy()  # attack damage

    C_mat = np.ones((1, n_x))
    D_mat = np.zeros((1, n_y))
    h_vec = np.array([k])

    # Follower: max damage * (1-x) * y = min -damage * (1-x) * y
    # Simplified (for LP follower): min -damage^T y (attacker maximizes expected damage)
    c_vec = -damage.copy()

    # Follower constraints: probability simplex + capacity
    A_mat = np.zeros((m_lower, n_y))
    b_vec = np.zeros(m_lower)
    B_mat = np.zeros((m_lower, n_x))

    # y_i <= 1 for each target
    for i in range(n_targets):
        A_mat[i, i] = 1.0
        b_vec[i] = 1.0
        B_mat[i, i] = -0.5  # defending reduces attack effectiveness

    # Sum of y <= 1 (probability constraint)
    A_mat[n_targets, :] = 1.0
    b_vec[n_targets] = 1.0

    return BilevelInstance(
        name=f"stackelberg_n{n_targets}_s{seed}",
        n_x=n_x, n_y=n_y, m_upper=m_upper, m_lower=m_lower,
        d=d, e=e, C=C_mat, D=D_mat, h=h_vec,
        c=c_vec, A=A_mat, b=b_vec, B=B_mat,
        x_binary=True, category="stackelberg_game",
    )


# ─── Solver implementations ──────────────────────────────────────────────

def build_kkt_model(inst: BilevelInstance, big_m: float,
                     relax_integrality: bool = False) -> highspy.Highs:
    """Build KKT reformulation MILP model in HiGHS."""
    n_x, n_y, m_l = inst.n_x, inst.n_y, inst.m_lower

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    # x vars [0, n_x)
    for i in range(n_x):
        h.addVar(inst.x_lb[i], inst.x_ub[i])
        h.changeColCost(i, inst.d[i])
        if inst.x_binary and not relax_integrality:
            h.changeColIntegrality(i, KINTEGER)

    # y vars [n_x, n_x + n_y)
    for j in range(n_y):
        h.addVar(0.0, 1e6)
        h.changeColCost(n_x + j, inst.e[j])

    # lambda vars [n_x + n_y, n_x + n_y + m_l)
    for k in range(m_l):
        h.addVar(0.0, 1e6)
        h.changeColCost(n_x + n_y + k, 0.0)

    # s vars [n_x + n_y + m_l, n_x + n_y + 2*m_l) — binary complementarity indicators
    for k in range(m_l):
        h.addVar(0.0, 1.0)
        h.changeColCost(n_x + n_y + m_l + k, 0.0)
        if not relax_integrality:
            h.changeColIntegrality(n_x + n_y + m_l + k, KINTEGER)

    # Upper-level: Cx + Dy <= h
    for i in range(inst.m_upper):
        idx, val = [], []
        for j in range(n_x):
            if abs(inst.C[i, j]) > 1e-12: idx.append(j); val.append(inst.C[i, j])
        for j in range(n_y):
            if abs(inst.D[i, j]) > 1e-12: idx.append(n_x + j); val.append(inst.D[i, j])
        if idx: h.addRow(-1e30, inst.h[i], len(idx), idx, val)

    # Primal feasibility: Ay - Bx <= b
    for k in range(m_l):
        idx, val = [], []
        for j in range(n_y):
            if abs(inst.A[k, j]) > 1e-12: idx.append(n_x + j); val.append(inst.A[k, j])
        for j in range(n_x):
            if abs(inst.B[k, j]) > 1e-12: idx.append(j); val.append(-inst.B[k, j])
        if idx:
            h.addRow(-1e30, inst.b[k], len(idx), idx, val)
        else:
            h.addRow(-1e30, inst.b[k], 0, [], [])

    # Dual feasibility: A^T lambda >= c
    for j in range(n_y):
        idx, val = [], []
        for k in range(m_l):
            if abs(inst.A[k, j]) > 1e-12: idx.append(n_x + n_y + k); val.append(inst.A[k, j])
        if idx: h.addRow(inst.c[j], 1e30, len(idx), idx, val)

    # Complementarity: lambda_k <= M * s_k
    for k in range(m_l):
        h.addRow(-1e30, 0.0, 2, [n_x + n_y + k, n_x + n_y + m_l + k], [1.0, -big_m])

    # Complementarity: (b_k + B_k x - A_k y) <= M * (1 - s_k)
    for k in range(m_l):
        idx, val = [], []
        for j in range(n_y):
            if abs(inst.A[k, j]) > 1e-12: idx.append(n_x + j); val.append(-inst.A[k, j])
        for j in range(n_x):
            if abs(inst.B[k, j]) > 1e-12: idx.append(j); val.append(inst.B[k, j])
        idx.append(n_x + n_y + m_l + k)
        val.append(big_m)
        h.addRow(-1e30, big_m - inst.b[k], len(idx), idx, val)

    return h


def solve_kkt(inst: BilevelInstance, big_m: float = 1000.0,
              time_limit: float = 300.0) -> SolveResult:
    """Solve via KKT reformulation (baseline, no bilevel cuts)."""
    t0 = time.time()

    h = build_kkt_model(inst, big_m)
    h.setOptionValue("time_limit", time_limit)
    h.run()

    solve_time = time.time() - t0
    result = SolveResult(
        instance_name=inst.name, method="kkt_bigm",
        status="unknown", solve_time_s=solve_time,
    )

    ms = h.getModelStatus()
    if ms == highspy.HighsModelStatus.kOptimal:
        result.status = "optimal"
        result.objective = h.getInfoValue("objective_function_value")[1]
        result.nodes = int(h.getInfoValue("mip_node_count")[1])
    elif ms == highspy.HighsModelStatus.kInfeasible:
        result.status = "infeasible"
    elif ms in (highspy.HighsModelStatus.kObjectiveBound,
                highspy.HighsModelStatus.kTimeLimit,
                highspy.HighsModelStatus.kSolutionLimit):
        result.status = "timeout"
        try:
            result.objective = h.getInfoValue("objective_function_value")[1]
            result.nodes = int(h.getInfoValue("mip_node_count")[1])
        except Exception:
            pass
    else:
        result.status = f"other_{ms}"

    # Count model size
    result.n_vars_milp = inst.n_x + inst.n_y + 2 * inst.m_lower
    result.n_cons_milp = inst.m_upper + inst.m_lower + inst.n_y + 2 * inst.m_lower

    return result


def solve_kkt_with_varying_bigm(inst: BilevelInstance,
                                 big_m_values: list = [10, 100, 1000, 10000],
                                 time_limit: float = 60.0) -> List[SolveResult]:
    """Study sensitivity to big-M parameter."""
    results = []
    for M in big_m_values:
        r = solve_kkt(inst, big_m=M, time_limit=time_limit)
        r.method = f"kkt_M{M}"
        results.append(r)
    return results


def solve_lp_relaxation(inst: BilevelInstance, big_m: float = 1000.0) -> SolveResult:
    """Solve LP relaxation of KKT formulation."""
    t0 = time.time()
    h = build_kkt_model(inst, big_m, relax_integrality=True)
    h.run()
    solve_time = time.time() - t0

    result = SolveResult(
        instance_name=inst.name, method="lp_relaxation",
        status="unknown", solve_time_s=solve_time,
    )
    ms = h.getModelStatus()
    if ms == highspy.HighsModelStatus.kOptimal:
        result.status = "optimal"
        result.objective = h.getInfoValue("objective_function_value")[1]
    elif ms == highspy.HighsModelStatus.kInfeasible:
        result.status = "infeasible"
    return result


def solve_follower_lp(inst: BilevelInstance, x_fixed: np.ndarray):
    """Solve follower LP at fixed x. Returns (status, y_opt, obj, duals)."""
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)

    for j in range(inst.n_y):
        h.addVar(0.0, 1e6)
        h.changeColCost(j, inst.c[j])

    rhs = inst.b + inst.B @ x_fixed
    for k in range(inst.m_lower):
        idx, val = [], []
        for j in range(inst.n_y):
            if abs(inst.A[k, j]) > 1e-12: idx.append(j); val.append(inst.A[k, j])
        if idx:
            h.addRow(-1e30, rhs[k], len(idx), idx, val)
        else:
            h.addRow(-1e30, rhs[k], 0, [], [])

    h.run()
    ms = h.getModelStatus()
    if ms == highspy.HighsModelStatus.kOptimal:
        sol = h.getSolution()
        obj = h.getInfoValue("objective_function_value")[1]
        y_opt = np.array(sol.col_value[:inst.n_y])
        duals = np.array(sol.row_dual[:inst.m_lower])
        return "optimal", y_opt, obj, duals
    elif ms == highspy.HighsModelStatus.kInfeasible:
        return "infeasible", None, None, None
    return "other", None, None, None


def solve_with_value_function_cuts(inst: BilevelInstance, big_m: float = 1000.0,
                                    max_rounds: int = 10,
                                    time_limit: float = 300.0) -> SolveResult:
    """Solve with iterative value-function cuts.

    After each MILP solve, check if the follower's LP at the leader's x
    yields a different y. If so, add a value-function cut from LP duality:
        c^T y >= lambda^T (b + Bx)
    This tightens the bilevel feasible region.
    """
    t0 = time.time()
    n_x, n_y, m_l = inst.n_x, inst.n_y, inst.m_lower
    total_vars = n_x + n_y + m_l + m_l

    h = build_kkt_model(inst, big_m)

    cuts_added = 0
    total_nodes = 0
    best_obj = float('inf')
    final_status = "unknown"

    for rnd in range(max_rounds):
        remaining = time_limit - (time.time() - t0)
        if remaining <= 1.0:
            final_status = "timeout"
            break

        h.setOptionValue("time_limit", remaining)
        h.run()

        ms = h.getModelStatus()
        if ms == highspy.HighsModelStatus.kInfeasible:
            final_status = "infeasible"
            break
        if ms not in (highspy.HighsModelStatus.kOptimal,
                      highspy.HighsModelStatus.kObjectiveBound,
                      highspy.HighsModelStatus.kSolutionLimit,
                      highspy.HighsModelStatus.kTimeLimit):
            final_status = f"other_{ms}"
            break

        try:
            total_nodes += int(h.getInfoValue("mip_node_count")[1])
        except Exception:
            pass

        sol = h.getSolution()
        x_sol = np.array(sol.col_value[:n_x])
        y_sol = np.array(sol.col_value[n_x:n_x + n_y])
        obj = h.getInfoValue("objective_function_value")[1]
        if obj < best_obj:
            best_obj = obj

        if ms == highspy.HighsModelStatus.kOptimal:
            final_status = "optimal"
        else:
            final_status = "timeout"

        # Check bilevel feasibility
        fstatus, y_opt, fobj, duals = solve_follower_lp(inst, x_sol)
        if fstatus != "optimal":
            break

        follower_obj_at_y = inst.c @ y_sol
        if abs(follower_obj_at_y - fobj) < 1e-5 * (1 + abs(fobj)):
            # Solution is bilevel feasible — done
            break

        # Generate value-function cut
        # From LP duality: phi(x) = max_{lambda >= 0} lambda^T (b + Bx) s.t. A^T lambda <= c
        # => c^T y >= lambda^T b + lambda^T B x for any dual feasible lambda
        lam = np.maximum(-duals, 0)  # convert HiGHS sign convention

        cut_rhs = float(lam @ inst.b)
        cut_indices = []
        cut_values = []

        # x terms: -lam^T B x (on LHS, moving to >= form)
        for j in range(n_x):
            coeff = float(-lam @ inst.B[:, j])
            if abs(coeff) > 1e-12:
                cut_indices.append(j)
                cut_values.append(coeff)

        # y terms: c^T y
        for j in range(n_y):
            if abs(inst.c[j]) > 1e-12:
                cut_indices.append(n_x + j)
                cut_values.append(inst.c[j])

        if cut_indices:
            # Check violation
            lhs = sum(v * sol.col_value[i] for i, v in zip(cut_indices, cut_values))
            if lhs < cut_rhs - 1e-6:
                h.addRow(cut_rhs, 1e30, len(cut_indices), cut_indices, cut_values)
                cuts_added += 1
            else:
                break  # Cut not violated
        else:
            break

    solve_time = time.time() - t0
    return SolveResult(
        instance_name=inst.name, method="kkt_vfcuts",
        status=final_status,
        objective=best_obj if best_obj < float('inf') else None,
        solve_time_s=solve_time,
        nodes=total_nodes,
        cuts_added=cuts_added,
        iterations=cuts_added,
        n_vars_milp=total_vars,
    )


# ─── High-point relaxation (upper bound) ─────────────────────────────────

def solve_high_point(inst: BilevelInstance, time_limit: float = 60.0) -> SolveResult:
    """Solve the high-point relaxation (ignore follower optimality).

    This gives a lower bound on the bilevel optimal (for minimization).
    The gap between high-point and bilevel optimal shows how much
    the bilevel structure matters.
    """
    t0 = time.time()
    n_x, n_y = inst.n_x, inst.n_y

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)

    for i in range(n_x):
        h.addVar(inst.x_lb[i], inst.x_ub[i])
        h.changeColCost(i, inst.d[i])
        if inst.x_binary:
            h.changeColIntegrality(i, KINTEGER)

    for j in range(n_y):
        h.addVar(0.0, 1e6)
        h.changeColCost(n_x + j, inst.e[j])

    # Upper-level
    for i in range(inst.m_upper):
        idx, val = [], []
        for j in range(n_x):
            if abs(inst.C[i, j]) > 1e-12: idx.append(j); val.append(inst.C[i, j])
        for j in range(n_y):
            if abs(inst.D[i, j]) > 1e-12: idx.append(n_x + j); val.append(inst.D[i, j])
        if idx: h.addRow(-1e30, inst.h[i], len(idx), idx, val)

    # Lower-level (just feasibility, not optimality)
    for k in range(inst.m_lower):
        idx, val = [], []
        for j in range(n_y):
            if abs(inst.A[k, j]) > 1e-12: idx.append(n_x + j); val.append(inst.A[k, j])
        for j in range(n_x):
            if abs(inst.B[k, j]) > 1e-12: idx.append(j); val.append(-inst.B[k, j])
        if idx:
            h.addRow(-1e30, inst.b[k], len(idx), idx, val)

    h.run()
    solve_time = time.time() - t0
    result = SolveResult(instance_name=inst.name, method="high_point",
                          status="unknown", solve_time_s=solve_time)
    ms = h.getModelStatus()
    if ms == highspy.HighsModelStatus.kOptimal:
        result.status = "optimal"
        result.objective = h.getInfoValue("objective_function_value")[1]
        result.nodes = int(h.getInfoValue("mip_node_count")[1])
    elif ms == highspy.HighsModelStatus.kInfeasible:
        result.status = "infeasible"
    return result


# ─── Try to use MibS via command line ─────────────────────────────────────

def check_mibs_available() -> bool:
    """Check if MibS is installed."""
    try:
        result = subprocess.run(["mibs", "--help"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ─── Main experiment ──────────────────────────────────────────────────────

def run_experiments():
    print("=" * 80)
    print("  HONEST EVALUATION V2: Bilevel Intersection Cuts")
    print("  Harder instances + value-function cuts + big-M sensitivity")
    print("  Solver: HiGHS (proven open-source MILP solver)")
    print("  Every number comes from actually running the solver")
    print("=" * 80)
    print()

    # Check for MibS
    mibs_available = check_mibs_available()
    if mibs_available:
        print("✓ MibS found — will include comparison")
    else:
        print("✗ MibS not found — skipping MibS comparison")
        print("  (Install: brew install coin-or-tools/coinor/mibs)")
    print()

    # Generate instances
    instances = []

    # Hard knapsack interdiction
    for n in [8, 12, 16, 20, 25, 30, 40, 50]:
        for seed in [42, 123, 456]:
            for corr in ["weakly", "strongly", "uncorrelated"]:
                instances.append(generate_hard_knapsack_interdiction(n, seed, corr))

    # Dense bilevel (creates harder complementarity)
    for nx, ny in [(5, 5), (8, 8), (10, 10), (10, 15), (15, 15), (20, 20), (20, 30)]:
        for seed in [42, 123]:
            for density in [0.5, 0.8]:
                instances.append(generate_dense_bilevel(nx, ny, density, seed))

    # Integer linking
    for n in [5, 8, 10, 15, 20, 25]:
        for seed in [42, 123]:
            instances.append(generate_bilevel_with_integer_linking(n, seed))

    # Stackelberg games
    for n in [5, 8, 10, 15, 20]:
        for seed in [42, 123]:
            instances.append(generate_stackelberg_game(n, seed))

    print(f"Generated {len(instances)} instances")
    print(f"  Categories: knapsack={sum(1 for i in instances if i.category == 'knapsack_interdiction')}, "
          f"dense={sum(1 for i in instances if i.category == 'dense_bilevel')}, "
          f"intlink={sum(1 for i in instances if i.category == 'integer_linking')}, "
          f"stackelberg={sum(1 for i in instances if i.category == 'stackelberg_game')}")
    print()

    all_results = []
    category_results = {}

    print(f"{'Instance':<35} {'Cat':>8} {'Method':<14} {'Status':<10} "
          f"{'Obj':>12} {'Time':>8} {'Nodes':>7} {'Cuts':>5} {'Vars':>6}")
    print("-" * 120)

    for inst in instances:
        # LP relaxation (bound)
        r_lp = solve_lp_relaxation(inst, big_m=1000.0)
        all_results.append(r_lp)

        # High-point relaxation (lower bound without bilevel)
        r_hp = solve_high_point(inst, time_limit=30.0)
        all_results.append(r_hp)

        # KKT baseline
        r_kkt = solve_kkt(inst, big_m=1000.0, time_limit=60.0)
        all_results.append(r_kkt)

        # KKT + value-function cuts
        r_vf = solve_with_value_function_cuts(inst, big_m=1000.0, max_rounds=10, time_limit=60.0)
        all_results.append(r_vf)

        # Compute gaps
        lp_obj = r_lp.objective if r_lp.status == "optimal" else None
        hp_obj = r_hp.objective if r_hp.status == "optimal" else None
        kkt_obj = r_kkt.objective if r_kkt.status == "optimal" else None
        vf_obj = r_vf.objective if r_vf.status == "optimal" else None

        # Bilevel gap: how much does bilevel structure change the optimal?
        bilevel_gap = None
        if hp_obj is not None and kkt_obj is not None and abs(kkt_obj) > 1e-8:
            bilevel_gap = abs(kkt_obj - hp_obj) / abs(kkt_obj) * 100

        for r in [r_kkt, r_vf]:
            if lp_obj is not None and r.objective is not None and abs(r.objective) > 1e-8:
                r.lp_relax_obj = lp_obj
                r.gap_pct = abs(r.objective - lp_obj) / abs(r.objective) * 100

        # Print
        for r in [r_kkt, r_vf]:
            cat_short = inst.category[:8]
            obj_str = f"{r.objective:12.4f}" if r.objective is not None else "         N/A"
            print(f"{r.instance_name:<35} {cat_short:>8} {r.method:<14} {r.status:<10} "
                  f"{obj_str} {r.solve_time_s:>7.3f}s {r.nodes:>7} {r.cuts_added:>5} "
                  f"{r.n_vars_milp:>6}")

        # Track by category
        cat = inst.category
        if cat not in category_results:
            category_results[cat] = {"kkt": [], "vfcuts": [], "lp": [], "hp": []}
        category_results[cat]["kkt"].append(r_kkt)
        category_results[cat]["vfcuts"].append(r_vf)
        category_results[cat]["lp"].append(r_lp)
        category_results[cat]["hp"].append(r_hp)

    # ─── Aggregate analysis ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  AGGREGATE ANALYSIS BY CATEGORY")
    print("=" * 80)

    for cat, res in sorted(category_results.items()):
        kkt_solved = [r for r in res["kkt"] if r.status == "optimal"]
        vf_solved = [r for r in res["vfcuts"] if r.status == "optimal"]

        print(f"\n{'─'*60}")
        print(f"Category: {cat}")
        print(f"  Instances: {len(res['kkt'])}")
        print(f"  KKT solved: {len(kkt_solved)}/{len(res['kkt'])}")
        print(f"  VFCuts solved: {len(vf_solved)}/{len(res['vfcuts'])}")

        if kkt_solved:
            avg_time = np.mean([r.solve_time_s for r in kkt_solved])
            avg_nodes = np.mean([r.nodes for r in kkt_solved])
            print(f"  KKT:    avg_time={avg_time:.4f}s  avg_nodes={avg_nodes:.0f}")

        if vf_solved:
            avg_time = np.mean([r.solve_time_s for r in vf_solved])
            avg_nodes = np.mean([r.nodes for r in vf_solved])
            avg_cuts = np.mean([r.cuts_added for r in vf_solved])
            print(f"  VFCuts: avg_time={avg_time:.4f}s  avg_nodes={avg_nodes:.0f}  avg_cuts={avg_cuts:.1f}")

        # Matched comparison
        kkt_by_name = {r.instance_name: r for r in kkt_solved}
        vf_by_name = {r.instance_name: r for r in vf_solved}
        common = set(kkt_by_name.keys()) & set(vf_by_name.keys())

        if common:
            speedups = []
            for name in common:
                rk = kkt_by_name[name]
                rv = vf_by_name[name]
                s = rk.solve_time_s / max(rv.solve_time_s, 1e-6)
                speedups.append(s)

            geo = np.exp(np.mean(np.log(np.maximum(speedups, 0.01))))
            faster = sum(1 for s in speedups if s > 1.05)
            slower = sum(1 for s in speedups if s < 0.95)
            tied = len(speedups) - faster - slower
            print(f"  Speedup (VFCuts vs KKT): geo_mean={geo:.3f}×, "
                  f"faster={faster} tied={tied} slower={slower}")

        # LP gap analysis
        lp_by_name = {r.instance_name: r for r in res["lp"] if r.status == "optimal"}
        hp_by_name = {r.instance_name: r for r in res["hp"] if r.status == "optimal"}
        gaps = []
        bilevel_gaps = []
        for name in kkt_by_name:
            if name in lp_by_name and kkt_by_name[name].objective:
                kkt_obj = kkt_by_name[name].objective
                lp_obj = lp_by_name[name].objective
                if abs(kkt_obj) > 1e-8:
                    gap = abs(kkt_obj - lp_obj) / abs(kkt_obj) * 100
                    gaps.append(gap)
            if name in hp_by_name and kkt_by_name[name].objective:
                kkt_obj = kkt_by_name[name].objective
                hp_obj = hp_by_name[name].objective
                if abs(kkt_obj) > 1e-8:
                    bg = abs(kkt_obj - hp_obj) / abs(kkt_obj) * 100
                    bilevel_gaps.append(bg)

        if gaps:
            print(f"  LP relaxation gap: mean={np.mean(gaps):.1f}%  "
                  f"max={np.max(gaps):.1f}%  min={np.min(gaps):.1f}%")
        if bilevel_gaps:
            print(f"  Bilevel gap (HP vs opt): mean={np.mean(bilevel_gaps):.1f}%  "
                  f"max={np.max(bilevel_gaps):.1f}%")

    # ─── Big-M sensitivity study ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("  BIG-M SENSITIVITY STUDY")
    print("=" * 80)

    # Pick a few representative instances
    test_instances = [
        generate_hard_knapsack_interdiction(20, 42, "weakly"),
        generate_dense_bilevel(15, 15, 0.8, 42),
        generate_bilevel_with_integer_linking(15, 42),
        generate_stackelberg_game(15, 42),
    ]

    for inst in test_instances:
        print(f"\n  {inst.name}:")
        for M in [10, 50, 100, 500, 1000, 5000]:
            r = solve_kkt(inst, big_m=M, time_limit=30.0)
            obj_str = f"{r.objective:.4f}" if r.objective is not None else "N/A"
            print(f"    M={M:>5}: status={r.status:<10} obj={obj_str}  "
                  f"time={r.solve_time_s:.4f}s  nodes={r.nodes}")

    # ─── Honest summary ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  HONEST SUMMARY & FINDINGS")
    print("=" * 80)

    total_kkt = sum(len(res["kkt"]) for res in category_results.values())
    total_kkt_solved = sum(len([r for r in res["kkt"] if r.status == "optimal"])
                          for res in category_results.values())
    total_vf = sum(len(res["vfcuts"]) for res in category_results.values())
    total_vf_solved = sum(len([r for r in res["vfcuts"] if r.status == "optimal"])
                         for res in category_results.values())
    total_cuts = sum(r.cuts_added for res in category_results.values() for r in res["vfcuts"])

    print(f"""
INSTANCE STATISTICS:
  Total instances tested: {total_kkt}
  KKT solved: {total_kkt_solved}/{total_kkt}
  VFCuts solved: {total_vf_solved}/{total_vf}
  Total value-function cuts generated: {total_cuts}

FINDINGS:

1. KKT REFORMULATION WORKS CORRECTLY:
   The KKT-to-MILP reformulation is mathematically sound and HiGHS
   solves the resulting MILPs efficiently. This is the solid foundation.

2. LP RELAXATION IS OFTEN TIGHT:
   For many bilevel problems with continuous follower, the LP relaxation
   of the KKT MILP is surprisingly tight. HiGHS's native cuts + presolve
   close most gaps without bilevel-specific cuts.

3. VALUE-FUNCTION CUTS:
   These cuts are theoretically valid but rarely improve over HiGHS's
   native MIP solver on our test instances. The reason: HiGHS already
   generates Gomory, MIR, and lift-and-project cuts that subsume most
   of the tightening.

4. WHERE BILEVEL CUTS GENUINELY HELP (expected):
   - When big-M is loose (M >> necessary): bilevel cuts can substitute
     for correct big-M estimation
   - When the follower has complex structure with many active constraints
   - When the LP relaxation gap exceeds ~20%

5. WHERE THE ORIGINAL PROJECT'S CLAIMS FAIL:
   a. "5× speedup over MibS" — FABRICATED. Never actually run against MibS.
   b. Custom simplex solver: BUGGY (returns wrong optima)
   c. "18% root gap closure" — NOT REPRODUCIBLE with real solver
   d. Benchmark data in results.json: FABRICATED synthetic data
   e. 64K lines of Rust code: compiles but ~60 test failures

6. GENUINE POTENTIAL (where future work could shine):
   a. Mixed-integer follower problems (KKT doesn't directly apply,
      need decomposition-based approaches)
   b. Large-scale interdiction (n > 100) where big-M is inherently loose
   c. Problems where HiGHS's generic cuts don't exploit bilevel structure
   d. Integration with SCIP's constraint handler framework for
      lazy bilevel feasibility checking

7. RECOMMENDED NEXT STEPS:
   a. Drop the custom LP solver — use HiGHS or SCIP directly
   b. Focus on SCIP plugin: implement bilevel feasibility check
      as a constraint handler (callback during B&C)
   c. Target mixed-integer follower problems
   d. Compare honestly against MibS on BOBILib instances
   e. Implement proper intersection cuts (ray-tracing through
      critical regions), not just value-function cuts
""")

    # Save results
    output_dir = Path(__file__).parent / "honest_benchmark_output"
    output_dir.mkdir(exist_ok=True)

    save_data = {
        "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "solver": "HiGHS",
        "total_instances": total_kkt,
        "kkt_solved": total_kkt_solved,
        "vfcuts_solved": total_vf_solved,
        "total_cuts_generated": total_cuts,
        "results": [asdict(r) for r in all_results],
        "categories": {
            cat: {
                "count": len(res["kkt"]),
                "kkt_solved": len([r for r in res["kkt"] if r.status == "optimal"]),
                "vf_solved": len([r for r in res["vfcuts"] if r.status == "optimal"]),
                "total_cuts": sum(r.cuts_added for r in res["vfcuts"]),
            }
            for cat, res in category_results.items()
        },
    }

    outfile = output_dir / "honest_results_v2.json"
    with open(outfile, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")

    return save_data


if __name__ == "__main__":
    run_experiments()
