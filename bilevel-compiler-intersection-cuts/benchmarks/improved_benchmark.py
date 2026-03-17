#!/usr/bin/env python3
"""
Improved BOBILib benchmark: testing whether better reformulations
actually fix the ~55% failure rate of naive Big-M.

Methods:
  1. BigM+SCIP (baseline): M=10^4, from previous run
  2. SOS1+SCIP: Exact complementarity via SOS1 constraints (no Big-M)
  3. Indicator+SCIP: Exact complementarity via indicator constraints
  4. TightM+SCIP: LP-based Big-M tightening before solving
  5. VFCuts+SCIP: Iterative value-function cutting planes
"""

import gzip
import json
import os
import sys
import time
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# MPS Parser
# ---------------------------------------------------------------------------

def parse_mps_gz(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='replace') as f:
        lines = f.readlines()

    rows, row_sense, obj_name = [], {}, ''
    cols, seen_cols = [], set()
    A, rhs, lo, hi = {}, {}, {}, {}
    integers = set()
    section, in_integer = None, False

    for raw in lines:
        line = raw.rstrip('\n')
        if not line or line[0] == '*':
            continue
        if line[0] != ' ':
            tok = line.split()[0]
            if tok in ('NAME','ROWS','COLUMNS','RHS','RANGES','BOUNDS','ENDATA'):
                section = tok
            continue
        parts = line.split()
        if section == 'ROWS' and len(parts) >= 2:
            rows.append(parts[1])
            row_sense[parts[1]] = parts[0]
            if parts[0] == 'N':
                obj_name = parts[1]
        elif section == 'COLUMNS':
            if "'MARKER'" in line:
                in_integer = "'INTORG'" in line
                continue
            if len(parts) < 3:
                continue
            col = parts[0]
            if col not in seen_cols:
                seen_cols.add(col)
                cols.append(col)
                lo[col] = 0.0
                hi[col] = 1e30
            if in_integer:
                integers.add(col)
            i = 1
            while i+1 < len(parts):
                A[(parts[i], col)] = float(parts[i+1])
                i += 2
        elif section == 'RHS' and len(parts) >= 3:
            i = 1
            while i+1 < len(parts):
                rhs[parts[i]] = float(parts[i+1])
                i += 2
        elif section == 'BOUNDS' and len(parts) >= 3:
            bt, _, col = parts[0], parts[1], parts[2]
            val = float(parts[3]) if len(parts) > 3 else 0.0
            if bt == 'UP': hi[col] = val
            elif bt == 'LO': lo[col] = val
            elif bt == 'FX': lo[col] = hi[col] = val
            elif bt == 'FR': lo[col] = -1e30; hi[col] = 1e30
            elif bt == 'MI': lo[col] = -1e30
            elif bt == 'PL': hi[col] = 1e30
            elif bt == 'BV': lo[col] = 0.0; hi[col] = 1.0; integers.add(col)

    return {'obj': obj_name, 'rows': rows, 'sense': row_sense,
            'cols': cols, 'A': A, 'rhs': rhs, 'lo': lo, 'hi': hi, 'int': integers}


def parse_aux(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    lower_vars, lower_obj, lower_constrs = [], {}, []
    sec = None
    for line in lines:
        l = line.strip()
        if not l: continue
        if l in ('@VARSBEGIN','@VARSEND','@CONSTRSBEGIN','@CONSTRSEND',
                 '@NUMVARS','@NUMCONSTRS','@NAME','@MPS'):
            sec = l; continue
        if sec == '@VARSBEGIN':
            p = l.split()
            lower_vars.append(p[0])
            lower_obj[p[0]] = float(p[1]) if len(p) > 1 else 0.0
        elif sec == '@CONSTRSBEGIN':
            lower_constrs.append(l.split()[0])
        elif sec in ('@NUMVARS','@NUMCONSTRS','@NAME','@MPS'):
            sec = None
    return lower_vars, lower_obj, lower_constrs


# ---------------------------------------------------------------------------
# Extract SCIP result
# ---------------------------------------------------------------------------

def _extract_scip(m, t0):
    elapsed = time.time() - t0
    r = {'time': elapsed, 'nodes': 0}
    try:
        r['nodes'] = int(m.getNNodes())
    except: pass
    st = m.getStatus()
    if st == 'optimal':
        r['status'] = 'optimal'
        r['objective'] = m.getObjVal()
    elif st == 'infeasible':
        r['status'] = 'infeasible'
        r['objective'] = None
    elif st in ('timelimit', 'gaplimit'):
        r['status'] = 'time_limit'
        try: r['objective'] = m.getObjVal()
        except: r['objective'] = None
    else:
        r['status'] = f'other:{st}'
        try: r['objective'] = m.getObjVal()
        except: r['objective'] = None
    return r


# ---------------------------------------------------------------------------
# Build base model (primal feasibility + stationarity, shared by all methods)
# ---------------------------------------------------------------------------

def build_base_model(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    """Build SCIP model with primal feasibility + dual feasibility (stationarity).
    Returns model, variables dict, and KKT metadata needed by complementarity methods."""
    from pyscipopt import Model, quicksum

    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    upper_constrs = [r for r in constraint_rows if r not in lower_constr_set]
    lower_rows = [r for r in constraint_rows if r in lower_constr_set]

    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in all_vars}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}

    m = Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", 1e-6)

    # Decision variables
    x = {}
    for v in all_vars:
        lb = max(mps['lo'].get(v, 0.0), -1e20)
        ub = min(mps['hi'].get(v, 1e30), 1e20)
        is_int = v in mps['int']
        vt = "I" if is_int else "C"
        x[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                         ub=ub if ub < 1e20 else None, vtype=vt)

    # Objective (upper level)
    m.setObjective(quicksum(c_upper.get(v, 0) * x[v] for v in all_vars
                             if abs(c_upper.get(v, 0)) > 1e-15), "minimize")

    # All constraints (primal feasibility)
    for row in constraint_rows:
        s = mps['sense'][row]
        rval = mps['rhs'].get(row, 0.0)
        lhs = quicksum(mps['A'].get((row, v), 0) * x[v] for v in all_vars
                        if abs(mps['A'].get((row, v), 0)) > 1e-15)
        if s == 'L': m.addCons(lhs <= rval, name=row)
        elif s == 'G': m.addCons(lhs >= rval, name=row)
        elif s == 'E': m.addCons(lhs == rval, name=row)

    if not lower_rows or not lower_vars:
        return m, x, None

    # Standardize lower constraints to <= form
    std_rows = []
    for row in lower_rows:
        s = mps['sense'][row]
        sign = -1.0 if s == 'G' else 1.0
        std_rows.append((row, sign, s))

    # Dual variables (lambda)
    lam = {}
    for k, (row, sign, s) in enumerate(std_rows):
        if s == 'E':
            lam[k] = m.addVar(f"lam_{k}", lb=None, ub=None, vtype="C")
        else:
            lam[k] = m.addVar(f"lam_{k}", lb=0, vtype="C")

    # Bound duals
    mu_lo, mu_hi = {}, {}
    for v in lower_vars:
        lb_v = mps['lo'].get(v, 0.0)
        ub_v = mps['hi'].get(v, 1e30)
        if lb_v > -1e20:
            mu_lo[v] = m.addVar(f"mu_lo_{v}", lb=0, vtype="C")
        if ub_v < 1e20:
            mu_hi[v] = m.addVar(f"mu_hi_{v}", lb=0, vtype="C")

    # Stationarity
    for v in lower_vars:
        grad = c_lower.get(v, 0.0)
        terms = []
        for k, (row, sign, _) in enumerate(std_rows):
            a = mps['A'].get((row, v), 0.0)
            if abs(a) < 1e-15: continue
            terms.append(lam[k] * (sign * a))
        stat_expr = grad + quicksum(terms) if terms else grad
        if v in mu_lo:
            stat_expr = stat_expr - mu_lo[v]
        if v in mu_hi:
            stat_expr = stat_expr + mu_hi[v]
        m.addCons(stat_expr == 0, name=f"stat_{v}")

    kkt_info = {
        'std_rows': std_rows,
        'lam': lam,
        'mu_lo': mu_lo,
        'mu_hi': mu_hi,
        'lower_vars': lower_vars,
        'lower_rows': lower_rows,
        'c_lower': c_lower,
        'all_vars': all_vars,
        'lower_var_set': lower_var_set,
    }
    return m, x, kkt_info


# ---------------------------------------------------------------------------
# Method 1: Big-M complementarity (baseline)
# ---------------------------------------------------------------------------

def solve_bigm(mps, lower_vars, lower_obj, lower_constrs, time_limit, big_m=1e4):
    from pyscipopt import quicksum
    m, x, kkt = build_base_model(mps, lower_vars, lower_obj, lower_constrs, time_limit)
    if kkt is None:
        t0 = time.time(); m.optimize(); return _extract_scip(m, t0)

    M = big_m
    std_rows = kkt['std_rows']
    lam = kkt['lam']
    mu_lo = kkt['mu_lo']
    mu_hi = kkt['mu_hi']
    all_vars = kkt['all_vars']

    # Constraint complementarity: lambda * slack = 0 via Big-M
    for k, (row, sign, s) in enumerate(std_rows):
        if s == 'E': continue
        z = m.addVar(f"z_{k}", vtype="B")
        rval = mps['rhs'].get(row, 0.0)
        m.addCons(lam[k] <= M * z, name=f"comp_d_{k}")
        slack = sign * (rval - quicksum(
            mps['A'].get((row, v), 0) * x[v] for v in all_vars
            if abs(mps['A'].get((row, v), 0)) > 1e-15
        ))
        m.addCons(slack <= M * (1 - z), name=f"comp_s_{k}")

    # Bound complementarity
    for v in kkt['lower_vars']:
        lb_v = mps['lo'].get(v, 0.0)
        ub_v = mps['hi'].get(v, 1e30)
        if v in mu_lo:
            zl = m.addVar(f"zl_{v}", vtype="B")
            m.addCons(mu_lo[v] <= M * zl, name=f"comp_mulo_{v}")
            m.addCons(x[v] - lb_v <= M * (1 - zl), name=f"comp_lb_{v}")
        if v in mu_hi and ub_v < 1e20:
            zu = m.addVar(f"zu_{v}", vtype="B")
            m.addCons(mu_hi[v] <= M * zu, name=f"comp_muhi_{v}")
            m.addCons(ub_v - x[v] <= M * (1 - zu), name=f"comp_ub_{v}")

    t0 = time.time(); m.optimize(); return _extract_scip(m, t0)


# ---------------------------------------------------------------------------
# Method 2: SOS1 complementarity (exact, no Big-M)
# ---------------------------------------------------------------------------

def solve_sos1(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    from pyscipopt import quicksum
    m, x, kkt = build_base_model(mps, lower_vars, lower_obj, lower_constrs, time_limit)
    if kkt is None:
        t0 = time.time(); m.optimize(); return _extract_scip(m, t0)

    std_rows = kkt['std_rows']
    lam = kkt['lam']
    mu_lo = kkt['mu_lo']
    mu_hi = kkt['mu_hi']
    all_vars = kkt['all_vars']

    # For each complementarity pair (dual, slack), add SOS1 constraint
    # SOS1({lambda_k, slack_k}) means at most one can be nonzero — EXACT!

    # Introduce explicit slack variables for cleaner SOS1
    for k, (row, sign, s) in enumerate(std_rows):
        if s == 'E': continue
        rval = mps['rhs'].get(row, 0.0)
        # slack_k = sign * (b - a^T x) >= 0
        sk = m.addVar(f"slack_{k}", lb=0, vtype="C")
        slack_expr = sign * (rval - quicksum(
            mps['A'].get((row, v), 0) * x[v] for v in all_vars
            if abs(mps['A'].get((row, v), 0)) > 1e-15
        ))
        m.addCons(sk == slack_expr, name=f"slack_def_{k}")
        # SOS1: at most one of {lambda_k, slack_k} is nonzero
        m.addConsSOS1([lam[k], sk], weights=[1.0, 2.0])

    # Bound complementarity via SOS1
    for v in kkt['lower_vars']:
        lb_v = mps['lo'].get(v, 0.0)
        ub_v = mps['hi'].get(v, 1e30)
        if v in mu_lo:
            gap_lo = m.addVar(f"gap_lo_{v}", lb=0, vtype="C")
            m.addCons(gap_lo == x[v] - lb_v, name=f"gap_lo_def_{v}")
            m.addConsSOS1([mu_lo[v], gap_lo], weights=[1.0, 2.0])
        if v in mu_hi and ub_v < 1e20:
            gap_hi = m.addVar(f"gap_hi_{v}", lb=0, vtype="C")
            m.addCons(gap_hi == ub_v - x[v], name=f"gap_hi_def_{v}")
            m.addConsSOS1([mu_hi[v], gap_hi], weights=[1.0, 2.0])

    t0 = time.time(); m.optimize(); return _extract_scip(m, t0)


# ---------------------------------------------------------------------------
# Method 3: Indicator complementarity (exact, no Big-M)
# ---------------------------------------------------------------------------

def solve_indicator(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    from pyscipopt import quicksum
    m, x, kkt = build_base_model(mps, lower_vars, lower_obj, lower_constrs, time_limit)
    if kkt is None:
        t0 = time.time(); m.optimize(); return _extract_scip(m, t0)

    std_rows = kkt['std_rows']
    lam = kkt['lam']
    mu_lo = kkt['mu_lo']
    mu_hi = kkt['mu_hi']
    all_vars = kkt['all_vars']

    # Indicator: z=0 -> lambda=0, z=1 -> slack=0
    for k, (row, sign, s) in enumerate(std_rows):
        if s == 'E': continue
        rval = mps['rhs'].get(row, 0.0)
        z = m.addVar(f"z_{k}", vtype="B")
        # z=0 => lambda_k = 0
        m.addConsIndicator(lam[k] <= 0, z, activeone=False, name=f"ind_d_{k}")
        # z=1 => slack = 0, i.e., sign*(b - a^T x) <= 0
        slack_expr = sign * (rval - quicksum(
            mps['A'].get((row, v), 0) * x[v] for v in all_vars
            if abs(mps['A'].get((row, v), 0)) > 1e-15
        ))
        m.addConsIndicator(slack_expr <= 0, z, activeone=True, name=f"ind_s_{k}")

    # Bound complementarity via indicators
    for v in kkt['lower_vars']:
        lb_v = mps['lo'].get(v, 0.0)
        ub_v = mps['hi'].get(v, 1e30)
        if v in mu_lo:
            zl = m.addVar(f"zl_{v}", vtype="B")
            m.addConsIndicator(mu_lo[v] <= 0, zl, activeone=False, name=f"ind_mulo_{v}")
            m.addConsIndicator(x[v] - lb_v <= 0, zl, activeone=True, name=f"ind_lb_{v}")
        if v in mu_hi and ub_v < 1e20:
            zu = m.addVar(f"zu_{v}", vtype="B")
            m.addConsIndicator(mu_hi[v] <= 0, zu, activeone=False, name=f"ind_muhi_{v}")
            m.addConsIndicator(ub_v - x[v] <= 0, zu, activeone=True, name=f"ind_ub_{v}")

    t0 = time.time(); m.optimize(); return _extract_scip(m, t0)


# ---------------------------------------------------------------------------
# Method 4: Tight Big-M (LP-based bound computation)
# ---------------------------------------------------------------------------

def solve_tight_bigm(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    """Compute tight M values from LP relaxation, then use Big-M."""
    from pyscipopt import Model, quicksum

    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    lower_rows = [r for r in constraint_rows if r in lower_constr_set]

    # Phase 1: Compute tight bounds on slacks and duals from LP relaxation
    # Solve the LP relaxation to get variable bounds
    lp = Model()
    lp.hideOutput()
    lp.setParam("limits/time", 5.0)  # Quick LP

    xv = {}
    for v in all_vars:
        lb = max(mps['lo'].get(v, 0.0), -1e20)
        ub = min(mps['hi'].get(v, 1e30), 1e20)
        xv[v] = lp.addVar(name=v, lb=lb if lb > -1e20 else None,
                           ub=ub if ub < 1e20 else None, vtype="C")

    for row in constraint_rows:
        s = mps['sense'][row]
        rval = mps['rhs'].get(row, 0.0)
        lhs = quicksum(mps['A'].get((row, v), 0) * xv[v] for v in all_vars
                        if abs(mps['A'].get((row, v), 0)) > 1e-15)
        if s == 'L': lp.addCons(lhs <= rval)
        elif s == 'G': lp.addCons(lhs >= rval)
        elif s == 'E': lp.addCons(lhs == rval)

    # Compute slack bounds for each lower constraint
    slack_ubs = {}
    for row in lower_rows:
        s = mps['sense'][row]
        if s == 'E': continue
        sign = -1.0 if s == 'G' else 1.0
        rval = mps['rhs'].get(row, 0.0)
        slack_expr = sign * (rval - quicksum(
            mps['A'].get((row, v), 0) * xv[v] for v in all_vars
            if abs(mps['A'].get((row, v), 0)) > 1e-15
        ))
        # Maximize slack to get upper bound
        lp.setObjective(slack_expr, "maximize")
        lp.optimize()
        if lp.getStatus() == 'optimal':
            slack_ubs[row] = max(lp.getObjVal() * 1.1, 1.0)  # 10% safety margin
        else:
            slack_ubs[row] = 1e4  # fallback
        lp.freeTransform()

    # Now solve with tight Big-M
    m, x, kkt = build_base_model(mps, lower_vars, lower_obj, lower_constrs, time_limit)
    if kkt is None:
        t0 = time.time(); m.optimize(); return _extract_scip(m, t0)

    std_rows = kkt['std_rows']
    lam = kkt['lam']
    mu_lo = kkt['mu_lo']
    mu_hi = kkt['mu_hi']

    for k, (row, sign, s) in enumerate(std_rows):
        if s == 'E': continue
        z = m.addVar(f"z_{k}", vtype="B")
        rval = mps['rhs'].get(row, 0.0)

        M_slack = slack_ubs.get(row, 1e4)
        # For dual bound: use a moderate estimate (tight M helps here too)
        M_dual = max(M_slack * 10, 100)  # dual can be larger

        m.addCons(lam[k] <= M_dual * z, name=f"comp_d_{k}")
        slack = sign * (rval - quicksum(
            mps['A'].get((row, v), 0) * x[v] for v in kkt['all_vars']
            if abs(mps['A'].get((row, v), 0)) > 1e-15
        ))
        m.addCons(slack <= M_slack * (1 - z), name=f"comp_s_{k}")

    # Bound complementarity with tight M
    for v in kkt['lower_vars']:
        lb_v = mps['lo'].get(v, 0.0)
        ub_v = mps['hi'].get(v, 1e30)
        M_var = max(abs(ub_v - lb_v) * 1.1, 1.0) if ub_v < 1e20 and lb_v > -1e20 else 1e4
        if v in mu_lo:
            zl = m.addVar(f"zl_{v}", vtype="B")
            m.addCons(mu_lo[v] <= 1e4 * zl, name=f"comp_mulo_{v}")
            m.addCons(x[v] - lb_v <= M_var * (1 - zl), name=f"comp_lb_{v}")
        if v in mu_hi and ub_v < 1e20:
            zu = m.addVar(f"zu_{v}", vtype="B")
            m.addCons(mu_hi[v] <= 1e4 * zu, name=f"comp_muhi_{v}")
            m.addCons(ub_v - x[v] <= M_var * (1 - zu), name=f"comp_ub_{v}")

    t0 = time.time(); m.optimize(); return _extract_scip(m, t0)


# ---------------------------------------------------------------------------
# Method 5: Value function cuts (iterative Benders-like)
# ---------------------------------------------------------------------------

def solve_vf_cuts(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    """Iterative value function cutting plane algorithm.
    
    Instead of encoding complementarity, we:
    1. Solve master: min upper_obj s.t. primal feasibility + VF cuts
    2. Given x*, solve follower LP to get phi(x*) and subgradient
    3. Add cut: c^T y <= phi_approx(x)  (affine underestimator)
    4. Repeat until c^T y* = phi(x*)
    """
    from pyscipopt import Model, quicksum

    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    upper_vars = [v for v in all_vars if v not in lower_var_set]
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    upper_constrs_list = [r for r in constraint_rows if r not in lower_constr_set]
    lower_rows = [r for r in constraint_rows if r in lower_constr_set]

    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in all_vars}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}

    if not lower_rows or not lower_vars:
        # No bilevel structure, just solve
        m = Model(); m.hideOutput()
        m.setParam("limits/time", time_limit)
        xv = {}
        for v in all_vars:
            lb = max(mps['lo'].get(v, 0.0), -1e20)
            ub = min(mps['hi'].get(v, 1e30), 1e20)
            xv[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                              ub=ub if ub < 1e20 else None,
                              vtype="I" if v in mps['int'] else "C")
        m.setObjective(quicksum(c_upper.get(v, 0) * xv[v] for v in all_vars
                                 if abs(c_upper.get(v, 0)) > 1e-15), "minimize")
        for row in constraint_rows:
            s = mps['sense'][row]
            rval = mps['rhs'].get(row, 0.0)
            lhs = quicksum(mps['A'].get((row, v), 0) * xv[v] for v in all_vars
                            if abs(mps['A'].get((row, v), 0)) > 1e-15)
            if s == 'L': m.addCons(lhs <= rval)
            elif s == 'G': m.addCons(lhs >= rval)
            elif s == 'E': m.addCons(lhs == rval)
        t0 = time.time(); m.optimize(); return _extract_scip(m, t0)

    # Identify linking: which lower constraints involve upper variables?
    # Lower constraint: sum_j a_{ij} y_j + sum_j b_{ij} x_j {<=,>=,=} rhs_i
    # Parametric RHS: rhs_i(x) = rhs_i - sum_j b_{ij} x_j

    t0 = time.time()
    MAX_ITERS = 50
    TOL = 1e-4
    best_obj = None
    best_sol = None
    cuts = []  # list of (gradient_dict_over_x, constant, lower_obj_val)

    for iteration in range(MAX_ITERS):
        if time.time() - t0 > time_limit:
            break

        # --- Master problem ---
        master = Model()
        master.hideOutput()
        master.setParam("limits/time", max(time_limit - (time.time() - t0), 1.0))
        master.setParam("limits/gap", 1e-6)

        xv = {}
        for v in all_vars:
            lb = max(mps['lo'].get(v, 0.0), -1e20)
            ub = min(mps['hi'].get(v, 1e30), 1e20)
            is_int = v in mps['int']
            xv[v] = master.addVar(name=v, lb=lb if lb > -1e20 else None,
                                   ub=ub if ub < 1e20 else None,
                                   vtype="I" if is_int else "C")

        # Epigraph variable for follower objective
        eta = master.addVar("eta", lb=None, ub=None, vtype="C")

        # Upper objective + eta (we want follower to be optimal)
        master.setObjective(
            quicksum(c_upper.get(v, 0) * xv[v] for v in all_vars
                      if abs(c_upper.get(v, 0)) > 1e-15),
            "minimize"
        )

        # All constraints
        for row in constraint_rows:
            s = mps['sense'][row]
            rval = mps['rhs'].get(row, 0.0)
            lhs = quicksum(mps['A'].get((row, v), 0) * xv[v] for v in all_vars
                            if abs(mps['A'].get((row, v), 0)) > 1e-15)
            if s == 'L': master.addCons(lhs <= rval, name=row)
            elif s == 'G': master.addCons(lhs >= rval, name=row)
            elif s == 'E': master.addCons(lhs == rval, name=row)

        # Follower optimality: c^T y >= eta
        master.addCons(
            quicksum(c_lower.get(v, 0) * xv[v] for v in lower_vars
                      if abs(c_lower.get(v, 0)) > 1e-15) >= eta,
            name="vf_link"
        )

        # Add accumulated VF cuts: eta >= alpha^T x + beta
        for ci, (grad, const) in enumerate(cuts):
            cut_expr = quicksum(grad.get(v, 0) * xv[v] for v in upper_vars
                                 if abs(grad.get(v, 0)) > 1e-15) + const
            master.addCons(eta >= cut_expr, name=f"vf_cut_{ci}")

        master.optimize()

        if master.getStatus() not in ('optimal', 'gaplimit'):
            r = _extract_scip(master, t0)
            r['vf_iters'] = iteration + 1
            return r

        # Extract solution
        x_val = {v: master.getVal(xv[v]) for v in all_vars}
        eta_val = master.getVal(eta)
        follower_obj_at_sol = sum(c_lower.get(v, 0) * x_val.get(v, 0) for v in lower_vars)
        master_obj = master.getObjVal()

        # --- Follower subproblem: min c^T y s.t. A y {<=,>=,=} b(x*) ---
        sub = Model()
        sub.hideOutput()
        sub.setParam("limits/time", 5.0)

        yv = {}
        for v in lower_vars:
            lb = max(mps['lo'].get(v, 0.0), -1e20)
            ub = min(mps['hi'].get(v, 1e30), 1e20)
            yv[v] = sub.addVar(name=v, lb=lb if lb > -1e20 else None,
                                ub=ub if ub < 1e20 else None, vtype="C")

        sub.setObjective(quicksum(c_lower.get(v, 0) * yv[v] for v in lower_vars
                                   if abs(c_lower.get(v, 0)) > 1e-15), "minimize")

        # Lower constraints with x fixed
        sub_constrs = []
        for row in lower_rows:
            s = mps['sense'][row]
            rval = mps['rhs'].get(row, 0.0)
            # RHS adjusted for upper variable contribution
            rhs_adj = rval - sum(mps['A'].get((row, v), 0) * x_val.get(v, 0)
                                  for v in upper_vars
                                  if abs(mps['A'].get((row, v), 0)) > 1e-15)
            lhs = quicksum(mps['A'].get((row, v), 0) * yv[v] for v in lower_vars
                            if abs(mps['A'].get((row, v), 0)) > 1e-15)
            if s == 'L': c_ = sub.addCons(lhs <= rhs_adj, name=row)
            elif s == 'G': c_ = sub.addCons(lhs >= rhs_adj, name=row)
            elif s == 'E': c_ = sub.addCons(lhs == rhs_adj, name=row)
            sub_constrs.append((row, c_))

        sub.optimize()

        if sub.getStatus() not in ('optimal', 'gaplimit'):
            # Follower infeasible for this x — add feasibility cut or stop
            break

        phi_x = sub.getObjVal()  # value function at x*

        # Check convergence: is follower_obj == phi(x)?
        gap = abs(follower_obj_at_sol - phi_x) / max(1.0, abs(phi_x))
        if gap < TOL:
            # Converged! Current solution is bilevel-feasible
            if best_obj is None or master_obj < best_obj:
                best_obj = master_obj
                best_sol = x_val
            break

        # --- Generate value function cut ---
        # phi(x) >= phi(x*) + pi^T (b(x) - b(x*))
        # where pi are dual multipliers and b(x) = rhs - B*x
        # So: phi(x) >= phi(x*) - pi^T B (x - x*)
        #           = phi(x*) - pi^T B x + pi^T B x*
        #           = (phi(x*) + pi^T B x*) - pi^T B x
        # Cut: eta >= alpha^T x + beta  where
        #   alpha_v = -sum_k pi_k * B_{kv}   (for upper vars)
        #   beta = phi(x*) + sum_k pi_k * sum_v B_{kv} * x_v*

        # Get duals (SCIP provides them for LP relaxation at root)
        try:
            duals = {}
            for row, c_ in sub_constrs:
                try:
                    duals[row] = sub.getDualsolLinear(c_)
                except:
                    duals[row] = 0.0
        except:
            duals = {row: 0.0 for row, _ in sub_constrs}

        # Compute cut coefficients
        grad = {}  # gradient w.r.t. upper vars
        for v in upper_vars:
            coeff = 0.0
            for row in lower_rows:
                a = mps['A'].get((row, v), 0.0)
                if abs(a) < 1e-15: continue
                pi = duals.get(row, 0.0)
                coeff -= pi * a  # -pi^T * B column
            if abs(coeff) > 1e-15:
                grad[v] = coeff

        # beta = phi(x*) - alpha^T x*
        beta = phi_x - sum(grad.get(v, 0) * x_val.get(v, 0) for v in upper_vars)

        cuts.append((grad, beta))

        if best_obj is None or master_obj < best_obj:
            best_obj = master_obj
            best_sol = x_val

    # Final result
    r = {'time': time.time() - t0, 'nodes': 0, 'vf_iters': len(cuts) + 1}
    if best_obj is not None:
        r['status'] = 'optimal'
        r['objective'] = best_obj
    else:
        r['status'] = 'no_solution'
        r['objective'] = None
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bobilib_dir = os.path.join(script_dir, 'bobilib')
    list_file = os.path.join(bobilib_dir, 'instance_list.json')

    if not os.path.exists(list_file):
        print("ERROR: instance_list.json not found")
        sys.exit(1)

    with open(list_file) as f:
        instance_list = json.load(f)

    # Use the same 20 instances as baseline for fair comparison
    MAX_INSTANCES = 20
    instance_list = instance_list[:MAX_INSTANCES]

    TIME_LIMIT = 60.0

    methods = [
        ("BigM",      lambda mps, lv, lo, lc: solve_bigm(mps, lv, lo, lc, TIME_LIMIT)),
        ("SOS1",      lambda mps, lv, lo, lc: solve_sos1(mps, lv, lo, lc, TIME_LIMIT)),
        ("Indicator",  lambda mps, lv, lo, lc: solve_indicator(mps, lv, lo, lc, TIME_LIMIT)),
        ("TightM",    lambda mps, lv, lo, lc: solve_tight_bigm(mps, lv, lo, lc, TIME_LIMIT)),
        ("VFCuts",    lambda mps, lv, lo, lc: solve_vf_cuts(mps, lv, lo, lc, TIME_LIMIT)),
    ]

    print("=" * 110)
    print("BOBILib Improved Benchmark: 5 Reformulation Strategies (all SCIP)")
    print(f"Instances: {len(instance_list)} | Time limit: {TIME_LIMIT}s")
    print("=" * 110)

    all_results = []
    for idx, entry in enumerate(instance_list):
        name = entry['name']
        aux_path = entry['aux_path']
        known_obj = entry['obj']
        difficulty = entry['difficulty']

        mps_gz = aux_path.replace('.aux', '.mps.gz')
        try:
            mps = parse_mps_gz(mps_gz)
            lower_vars, lower_obj, lower_constrs = parse_aux(aux_path)
        except Exception as e:
            print(f"\n[{idx+1}] {name}: PARSE ERROR: {e}")
            continue

        n_total = len(mps['cols'])
        n_lower = len(lower_vars)
        n_lower_c = len(lower_constrs)

        print(f"\n[{idx+1}/{len(instance_list)}] {name} "
              f"(v={n_total} lv={n_lower} lc={n_lower_c} opt={known_obj})")

        row = {'name': name, 'n_vars': n_total, 'n_lower_vars': n_lower,
               'n_lower_constrs': n_lower_c,
               'difficulty': difficulty, 'known_obj': known_obj, 'methods': {}}

        for mname, fn in methods:
            try:
                r = fn(mps, lower_vars, lower_obj, lower_constrs)
            except Exception as e:
                r = {'status': 'crash', 'objective': None, 'time': 0, 'nodes': 0}
                if '--debug' in sys.argv:
                    traceback.print_exc()

            obj = r.get('objective')
            gap = None
            if obj is not None and known_obj is not None:
                if abs(known_obj) > 1e-8:
                    gap = abs(obj - known_obj) / max(1.0, abs(known_obj))
                else:
                    gap = abs(obj - known_obj)
            r['gap'] = gap

            row['methods'][mname] = r

            obj_s = f"{obj:.2f}" if obj is not None else "None"
            gap_s = f"{gap:.1e}" if gap is not None else "---"
            ok = "✓" if gap is not None and gap < 0.01 else ("✗" if gap is not None else "?")
            extra = f" vf_iters={r['vf_iters']}" if 'vf_iters' in r else ""
            print(f"  {mname:12s} {r['status']:12s} obj={obj_s:>12s} gap={gap_s:>9s} "
                  f"{ok} {r['time']:.2f}s{extra}")

        all_results.append(row)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    mnames = [m[0] for m in methods]
    N = len(all_results)

    print(f"\n{'Method':12s} {'Correct':>8s} {'Optimal':>8s} {'Crash':>8s} "
          f"{'AvgTime':>8s} {'MedTime':>8s} {'Overshoots':>11s}")
    print("-" * 80)
    for mn in mnames:
        cor = sum(1 for r in all_results
                  if r['methods'].get(mn,{}).get('gap') is not None
                  and r['methods'][mn]['gap'] < 0.01)
        opt = sum(1 for r in all_results if r['methods'].get(mn,{}).get('status') == 'optimal')
        crash = sum(1 for r in all_results if 'crash' in str(r['methods'].get(mn,{}).get('status','')))
        times = [r['methods'][mn]['time'] for r in all_results
                 if r['methods'].get(mn,{}).get('time') is not None]
        avg_t = f"{np.mean(times):.2f}" if times else "---"
        med_t = f"{np.median(times):.2f}" if times else "---"

        # Count overshoots (objective better than known optimal)
        overshoots = 0
        for r in all_results:
            obj = r['methods'].get(mn,{}).get('objective')
            ko = r['known_obj']
            if obj is not None and ko is not None:
                if obj < ko - 0.01 * max(1, abs(ko)):
                    overshoots += 1

        print(f"{mn:12s} {cor:>7d}/{N} {opt:>7d}/{N} {crash:>7d}/{N} "
              f"{avg_t:>8s} {med_t:>8s} {overshoots:>10d}/{N}")

    # Instance-by-instance comparison
    print(f"\n{'Instance':30s}", end="")
    for mn in mnames:
        print(f" {mn:>12s}", end="")
    print(f" {'Known':>10s}")
    print("-" * (30 + 12 * len(mnames) + 12))
    for r in all_results:
        print(f"{r['name']:30s}", end="")
        for mn in mnames:
            obj = r['methods'].get(mn, {}).get('objective')
            gap = r['methods'].get(mn, {}).get('gap')
            if obj is not None:
                ok = "✓" if gap is not None and gap < 0.01 else " "
                print(f" {obj:>10.1f}{ok}", end="")
            else:
                print(f" {'---':>11s}", end="")
        ko = r['known_obj']
        print(f" {ko:>10.1f}" if ko is not None else " ---")

    # Save
    out_dir = os.path.join(script_dir, 'bobilib_results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'improved_results.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
