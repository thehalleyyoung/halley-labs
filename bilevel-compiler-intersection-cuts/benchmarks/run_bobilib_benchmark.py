#!/usr/bin/env python3
"""
BOBILib benchmark: honest comparison of bilevel solving approaches using
real instances from the Bilevel Optimization Benchmark Instance Library.

Approaches compared (all open-source):
  1. KKT (LP relaxation) + SCIP
  2. Big-M complementarity + SCIP
  3. KKT (LP relaxation) + HiGHS
  4. Big-M complementarity + HiGHS

For integer lower-level problems, KKT is applied to the LP relaxation of
the lower level (HPR approach). This gives a valid relaxation but may not
yield the bilevel-optimal solution. Big-M similarly linearizes the
complementarity from the LP relaxation.

All results compared against known BOBILib optimal solutions.
"""

import gzip
import json
import os
import sys
import time
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# MPS Parser (fast, fixed-format)
# ---------------------------------------------------------------------------

def parse_mps_gz(path):
    """Parse gzipped MPS. Returns dict with problem data."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='replace') as f:
        lines = f.readlines()

    rows = []
    row_sense = {}
    obj_name = ''
    cols = []
    seen_cols = set()
    A = {}
    rhs = {}
    lo = {}
    hi = {}
    integers = set()
    section = None
    in_integer = False

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
            elif bt == 'BV':
                lo[col] = 0.0; hi[col] = 1.0; integers.add(col)

    return {'obj': obj_name, 'rows': rows, 'sense': row_sense,
            'cols': cols, 'A': A, 'rhs': rhs, 'lo': lo, 'hi': hi, 'int': integers}


def parse_aux(path):
    with open(path) as f:
        lines = f.read().strip().split('\n')
    lower_vars = []
    lower_obj = {}
    lower_constrs = []
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
# Solver wrappers
# ---------------------------------------------------------------------------

def solve_with_scip(mps, lower_vars, lower_obj, lower_constrs, method, time_limit, big_m=1e4):
    """Build reformulation and solve with SCIP."""
    from pyscipopt import Model, quicksum

    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)

    all_vars = mps['cols']
    var_idx = {v: i for i, v in enumerate(all_vars)}
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    upper_constrs = [r for r in constraint_rows if r not in lower_constr_set]
    lower_rows = [r for r in constraint_rows if r in lower_constr_set]

    # Upper-level objective
    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in all_vars}
    # Lower-level objective from aux
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}

    m = Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", 1e-6)

    # Variables
    x = {}
    for v in all_vars:
        lb = max(mps['lo'].get(v, 0.0), -1e20)
        ub = min(mps['hi'].get(v, 1e30), 1e20)
        # Upper-level integers stay integer; lower-level integers relaxed for KKT
        is_int = v in mps['int'] and v not in lower_var_set
        vt = "I" if is_int else "C"
        if v in mps['int'] and v in lower_var_set and method == 'bigm':
            vt = "I"  # Keep lower-level integrality for Big-M approach
        x[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                         ub=ub if ub < 1e20 else None, vtype=vt)

    # Objective
    m.setObjective(quicksum(c_upper.get(v, 0) * x[v] for v in all_vars
                             if abs(c_upper.get(v, 0)) > 1e-15), "minimize")

    # Helper to add constraint from MPS row
    def add_row(row, name_prefix=""):
        s = mps['sense'][row]
        rval = mps['rhs'].get(row, 0.0)
        lhs = quicksum(mps['A'].get((row, v), 0) * x[v] for v in all_vars
                        if abs(mps['A'].get((row, v), 0)) > 1e-15)
        nm = f"{name_prefix}{row}"
        if s == 'L': m.addCons(lhs <= rval, name=nm)
        elif s == 'G': m.addCons(lhs >= rval, name=nm)
        elif s == 'E': m.addCons(lhs == rval, name=nm)

    # Upper constraints
    for row in upper_constrs:
        add_row(row, "u_")

    # Lower primal feasibility
    for row in lower_rows:
        add_row(row, "l_")

    if not lower_rows or not lower_vars:
        m.optimize()
        return _extract_scip_result(m, None, None, method)

    # KKT conditions (applied to LP relaxation of lower level)
    # Convention: convert everything to standard form  a^T x <= b  internally.
    # For Ge rows: negate to get -a^T x <= -b.  Dual lambda >= 0 for each <= row.
    # For Eq rows: dual is free.
    # Store the standardized constraint data.
    std_rows = []  # (row, a_coeffs_sign, rhs_sign)  -- sign=1 for L, -1 for G
    lam = {}
    for k, row in enumerate(lower_rows):
        s = mps['sense'][row]
        sign = -1.0 if s == 'G' else 1.0  # negate Ge rows
        std_rows.append((row, sign))
        if s == 'E':
            lam[k] = m.addVar(f"lam_{k}", lb=None, ub=None, vtype="C")
        else:
            lam[k] = m.addVar(f"lam_{k}", lb=0, vtype="C")  # always >= 0 in standard form

    # Bound duals for lower-level variables
    mu_lo = {}  # dual for x >= lb  (nonneg)
    mu_hi = {}  # dual for x <= ub  (nonneg)
    for v in lower_vars:
        lb_v = mps['lo'].get(v, 0.0)
        ub_v = mps['hi'].get(v, 1e30)
        if lb_v > -1e20:
            mu_lo[v] = m.addVar(f"mu_lo_{v}", lb=0, vtype="C")
        if ub_v < 1e20:
            mu_hi[v] = m.addVar(f"mu_hi_{v}", lb=0, vtype="C")

    # Stationarity: c_j + sum_k lambda_k * (sign_k * a_{kj}) - mu_lo_j + mu_hi_j = 0
    for v in lower_vars:
        grad = c_lower.get(v, 0.0)
        terms = []
        for k, (row, sign) in enumerate(std_rows):
            a = mps['A'].get((row, v), 0.0)
            if abs(a) < 1e-15: continue
            terms.append(lam[k] * (sign * a))

        stat_expr = grad + quicksum(terms) if terms else grad
        if v in mu_lo:
            stat_expr = stat_expr - mu_lo[v]
        if v in mu_hi:
            stat_expr = stat_expr + mu_hi[v]
        m.addCons(stat_expr == 0, name=f"stat_{v}")

    if method == 'kkt':
        # Strong duality: c^T y = sum_k lambda_k * (sign_k * b_k) - sum mu_lo*lb + sum mu_hi*ub
        primal = quicksum(c_lower.get(v, 0) * x[v] for v in lower_vars
                           if abs(c_lower.get(v, 0)) > 1e-15)
        dual_terms = []
        for k, (row, sign) in enumerate(std_rows):
            rval = mps['rhs'].get(row, 0.0)
            dual_terms.append(lam[k] * (sign * rval))
        for v in lower_vars:
            lb_v = mps['lo'].get(v, 0.0)
            if v in mu_lo and abs(lb_v) > 1e-15:
                dual_terms.append(-mu_lo[v] * lb_v)
            ub_v = mps['hi'].get(v, 1e30)
            if v in mu_hi and ub_v < 1e20:
                dual_terms.append(mu_hi[v] * ub_v)
        dual = quicksum(dual_terms) if dual_terms else 0
        m.addCons(primal == dual, name="strong_duality")

        # Complementary slackness for bounds (via strong duality, already implicit)
        # mu_lo * (x - lb) = 0, mu_hi * (ub - x) = 0
        # These are nonlinear! Use SCIP's ability to handle them or linearize
        for v in lower_vars:
            lb_v = mps['lo'].get(v, 0.0)
            ub_v = mps['hi'].get(v, 1e30)
            if v in mu_lo:
                # mu_lo * (x - lb) = 0  -- SCIP handles bilinear via spatial B&B
                m.addCons(mu_lo[v] * (x[v] - lb_v) == 0, name=f"comp_lb_{v}")
            if v in mu_hi and ub_v < 1e20:
                m.addCons(mu_hi[v] * (ub_v - x[v]) == 0, name=f"comp_ub_{v}")

        # Complementary slackness for constraints
        for k, (row, sign) in enumerate(std_rows):
            s = mps['sense'][row]
            if s == 'E': continue
            rval = mps['rhs'].get(row, 0.0)
            slack = sign * (rval - quicksum(
                mps['A'].get((row, v), 0) * x[v] for v in all_vars
                if abs(mps['A'].get((row, v), 0)) > 1e-15
            ))
            m.addCons(lam[k] * slack == 0, name=f"comp_c_{k}")

    elif method == 'bigm':
        # Big-M linearization of complementary slackness
        M = big_m
        # Constraint complementarity
        for k, (row, sign) in enumerate(std_rows):
            s = mps['sense'][row]
            if s == 'E': continue

            z = m.addVar(f"z_{k}", vtype="B")
            rval = mps['rhs'].get(row, 0.0)

            # lambda_k <= M * z_k
            m.addCons(lam[k] <= M * z, name=f"comp_d_{k}")

            # slack_k = sign * (b - a^T x) <= M * (1 - z_k)
            slack = sign * (rval - quicksum(
                mps['A'].get((row, v), 0) * x[v] for v in all_vars
                if abs(mps['A'].get((row, v), 0)) > 1e-15
            ))
            m.addCons(slack <= M * (1 - z), name=f"comp_s_{k}")

        # Bound complementarity
        for v in lower_vars:
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

    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0

    return _extract_scip_result(m, elapsed, method)


def _extract_scip_result(m, elapsed, method):
    r = {'method': method, 'solver': 'scip', 'time': elapsed or 0, 'nodes': 0}
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


def solve_with_highs(mps, lower_vars, lower_obj, lower_constrs, method, time_limit, big_m=1e4):
    """Build Big-M reformulation and solve with HiGHS.
    HiGHS can only do linear/MIP, so KKT with bilinear complementarity
    is not supported. For 'kkt' method, we still use Big-M linearization."""
    import highspy

    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    upper_rows = [r for r in constraint_rows if r not in lower_constr_set]
    lower_rows_list = [r for r in constraint_rows if r in lower_constr_set]

    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in all_vars}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("mip_rel_gap", 1e-6)

    INF = highspy.kHighsInf
    col_idx = {}
    next_col = [0]

    def add_var(name, lb=0, ub=1e30, is_int=False):
        h.addVar(max(lb, -1e20), min(ub, 1e20))
        col_idx[name] = next_col[0]
        if is_int:
            h.changeColIntegrality(next_col[0], highspy._core.HighsVarType.kInteger)
        next_col[0] += 1

    def add_row(indices, vals, lb, ub):
        h.addRow(lb, ub, len(indices), indices, vals)

    # Original variables
    for v in all_vars:
        lb = mps['lo'].get(v, 0.0)
        ub = mps['hi'].get(v, 1e30)
        is_int = v in mps['int']
        add_var(v, lb, ub, is_int)

    # Objective
    for v in all_vars:
        c = c_upper.get(v, 0)
        if abs(c) > 1e-15:
            h.changeColCost(col_idx[v], c)

    # Upper constraints
    for row in upper_rows:
        s = mps['sense'][row]
        rval = mps['rhs'].get(row, 0.0)
        idxs, vals = [], []
        for v in all_vars:
            a = mps['A'].get((row, v), 0.0)
            if abs(a) > 1e-15:
                idxs.append(col_idx[v]); vals.append(a)
        if not idxs: continue
        if s == 'L': add_row(idxs, vals, -INF, rval)
        elif s == 'G': add_row(idxs, vals, rval, INF)
        elif s == 'E': add_row(idxs, vals, rval, rval)

    # Lower primal feasibility
    for row in lower_rows_list:
        s = mps['sense'][row]
        rval = mps['rhs'].get(row, 0.0)
        idxs, vals = [], []
        for v in all_vars:
            a = mps['A'].get((row, v), 0.0)
            if abs(a) > 1e-15:
                idxs.append(col_idx[v]); vals.append(a)
        if not idxs: continue
        if s == 'L': add_row(idxs, vals, -INF, rval)
        elif s == 'G': add_row(idxs, vals, rval, INF)
        elif s == 'E': add_row(idxs, vals, rval, rval)

    if not lower_rows_list or not lower_vars:
        t0 = time.time()
        h.run()
        return _extract_highs_result(h, time.time()-t0, method)

    # Standardize constraints to <= form
    std_rows = []
    M = big_m

    # Dual variables (lambda >= 0 in standard form)
    lam_idx = {}
    for k, row in enumerate(lower_rows_list):
        s = mps['sense'][row]
        sign = -1.0 if s == 'G' else 1.0
        std_rows.append((row, sign))
        if s == 'E':
            add_var(f"lam_{k}", -1e20, 1e20)
        else:
            add_var(f"lam_{k}", 0, 1e20)
        lam_idx[k] = col_idx[f"lam_{k}"]

    # Bound duals
    mu_lo_idx = {}
    mu_hi_idx = {}
    for v in lower_vars:
        lb_v = mps['lo'].get(v, 0.0)
        ub_v = mps['hi'].get(v, 1e30)
        if lb_v > -1e20:
            add_var(f"mu_lo_{v}", 0, 1e20)
            mu_lo_idx[v] = col_idx[f"mu_lo_{v}"]
        if ub_v < 1e20:
            add_var(f"mu_hi_{v}", 0, 1e20)
            mu_hi_idx[v] = col_idx[f"mu_hi_{v}"]

    # Stationarity: c_j + sum lambda_k * sign_k * a_kj - mu_lo_j + mu_hi_j = 0
    for v in lower_vars:
        grad = c_lower.get(v, 0.0)
        idxs, vals = [], []
        for k, (row, sign) in enumerate(std_rows):
            a = mps['A'].get((row, v), 0.0)
            if abs(a) < 1e-15: continue
            idxs.append(lam_idx[k]); vals.append(sign * a)
        if v in mu_lo_idx:
            idxs.append(mu_lo_idx[v]); vals.append(-1.0)
        if v in mu_hi_idx:
            idxs.append(mu_hi_idx[v]); vals.append(1.0)
        if idxs:
            add_row(idxs, vals, -grad, -grad)

    # Big-M complementarity for constraints
    for k, (row, sign) in enumerate(std_rows):
        s = mps['sense'][row]
        if s == 'E': continue
        add_var(f"z_{k}", 0, 1, True)
        zi = col_idx[f"z_{k}"]
        # lambda_k <= M * z_k
        add_row([lam_idx[k], zi], [1.0, -M], -INF, 0)
        # slack = sign*(b - a^T x) <= M*(1-z_k)
        rval = mps['rhs'].get(row, 0.0)
        sidx, sval = [], []
        for v in all_vars:
            a = mps['A'].get((row, v), 0.0)
            if abs(a) < 1e-15: continue
            sidx.append(col_idx[v]); sval.append(-sign * a)
        sidx.append(zi); sval.append(M)
        rhs_adj = M + sign * rval
        add_row(sidx, sval, -INF, rhs_adj)

    # Big-M complementarity for bounds
    for v in lower_vars:
        lb_v = mps['lo'].get(v, 0.0)
        if v in mu_lo_idx:
            add_var(f"zl_{v}", 0, 1, True)
            zli = col_idx[f"zl_{v}"]
            add_row([mu_lo_idx[v], zli], [1.0, -M], -INF, 0)
            add_row([col_idx[v], zli], [1.0, M], -INF, M + lb_v)  # x-lb <= M*(1-z)
        ub_v = mps['hi'].get(v, 1e30)
        if v in mu_hi_idx and ub_v < 1e20:
            add_var(f"zu_{v}", 0, 1, True)
            zui = col_idx[f"zu_{v}"]
            add_row([mu_hi_idx[v], zui], [1.0, -M], -INF, 0)
            add_row([col_idx[v], zui], [-1.0, M], -INF, M - ub_v)  # ub-x <= M*(1-z)

    t0 = time.time()
    h.run()
    return _extract_highs_result(h, time.time()-t0, method)


def _extract_highs_result(h, elapsed, method):
    import highspy
    r = {'method': method, 'solver': 'highs', 'time': elapsed, 'nodes': 0}
    try:
        r['nodes'] = int(h.getInfoValue("mip_node_count")[1])
    except: pass

    ms = h.getModelStatus()
    if ms == highspy.HighsModelStatus.kOptimal:
        r['status'] = 'optimal'
        r['objective'] = h.getInfoValue("objective_function_value")[1]
    elif ms == highspy.HighsModelStatus.kInfeasible:
        r['status'] = 'infeasible'
        r['objective'] = None
    elif ms in (highspy.HighsModelStatus.kObjectiveBound,
                highspy.HighsModelStatus.kTimeLimit):
        r['status'] = 'time_limit'
        try: r['objective'] = h.getInfoValue("objective_function_value")[1]
        except: r['objective'] = None
    else:
        r['status'] = f'other:{ms}'
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
        print("ERROR: Run bobilib_benchmark.py first to generate instance_list.json")
        sys.exit(1)

    with open(list_file) as f:
        instance_list = json.load(f)

    TIME_LIMIT = 60.0
    BIG_M = 1e4

    methods = [
        ("KKT+SCIP",  lambda mps, lv, lo, lc: solve_with_scip(mps, lv, lo, lc, 'kkt', TIME_LIMIT)),
        ("BigM+SCIP", lambda mps, lv, lo, lc: solve_with_scip(mps, lv, lo, lc, 'bigm', TIME_LIMIT, BIG_M)),
        ("BigM+HiGHS",lambda mps, lv, lo, lc: solve_with_highs(mps, lv, lo, lc, 'bigm', TIME_LIMIT, BIG_M)),
    ]

    print("=" * 90)
    print("BOBILib Benchmark: Open-Source Bilevel Solver Comparison")
    print(f"Instances: {len(instance_list)} | Time limit: {TIME_LIMIT}s | Big-M: {BIG_M}")
    print("=" * 90)

    all_results = []
    for idx, entry in enumerate(instance_list):
        name = entry['name']
        aux_path = entry['aux_path']
        sol_dir = entry['sol_dir']
        known_obj = entry['obj']
        difficulty = entry['difficulty']

        # Parse
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
        n_int = len(mps['int'])

        print(f"\n[{idx+1}/{len(instance_list)}] {name} "
              f"(v={n_total} lv={n_lower} lc={n_lower_c} int={n_int} "
              f"diff={difficulty} opt={known_obj})")

        row = {'name': name, 'n_vars': n_total, 'n_lower_vars': n_lower,
               'n_lower_constrs': n_lower_c, 'n_int': n_int,
               'difficulty': difficulty, 'known_obj': known_obj, 'methods': {}}

        for mname, fn in methods:
            try:
                r = fn(mps, lower_vars, lower_obj, lower_constrs)
            except Exception as e:
                r = {'status': f'crash', 'objective': None, 'time': 0, 'nodes': 0}
                # traceback.print_exc()

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
            correct = "✓" if gap is not None and gap < 0.01 else ("✗" if gap is not None else "?")
            print(f"  {mname:12s} {r['status']:12s} obj={obj_s:>12s} gap={gap_s:>9s} "
                  f"{correct} {r['time']:.2f}s {r.get('nodes',0):>6d}n")

        all_results.append(row)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    mnames = [m[0] for m in methods]
    N = len(all_results)

    print(f"\n{'Method':15s} {'Optimal':>8s} {'Correct':>8s} {'Infeas':>8s} {'Timeout':>8s} "
          f"{'Crash':>8s} {'AvgTime':>8s} {'MedTime':>8s}")
    print("-" * 85)
    for mn in mnames:
        opt = sum(1 for r in all_results if r['methods'].get(mn,{}).get('status') == 'optimal')
        cor = sum(1 for r in all_results
                  if r['methods'].get(mn,{}).get('gap') is not None
                  and r['methods'][mn]['gap'] < 0.01)
        inf = sum(1 for r in all_results if r['methods'].get(mn,{}).get('status') == 'infeasible')
        tl = sum(1 for r in all_results if 'time_limit' in str(r['methods'].get(mn,{}).get('status','')))
        crash = sum(1 for r in all_results if 'crash' in str(r['methods'].get(mn,{}).get('status','')))
        times = [r['methods'][mn]['time'] for r in all_results
                 if r['methods'].get(mn,{}).get('status') == 'optimal']
        avg_t = f"{np.mean(times):.2f}" if times else "---"
        med_t = f"{np.median(times):.2f}" if times else "---"
        print(f"{mn:15s} {opt:>7d}/{N} {cor:>7d}/{N} {inf:>7d}/{N} {tl:>7d}/{N} "
              f"{crash:>7d}/{N} {avg_t:>8s} {med_t:>8s}")

    # By difficulty
    for diff in ['easy', 'hard']:
        sub = [r for r in all_results if r['difficulty'] == diff]
        if not sub:
            continue
        Ns = len(sub)
        print(f"\n  {diff.upper()} ({Ns} instances):")
        for mn in mnames:
            opt = sum(1 for r in sub if r['methods'].get(mn,{}).get('status') == 'optimal')
            cor = sum(1 for r in sub
                      if r['methods'].get(mn,{}).get('gap') is not None
                      and r['methods'][mn]['gap'] < 0.01)
            print(f"    {mn:15s} optimal={opt:>3d}/{Ns}  correct={cor:>3d}/{Ns}")

    # Head-to-head
    print("\n--- Head-to-Head (instances both solved) ---")
    for i, m1 in enumerate(mnames):
        for m2 in mnames[i+1:]:
            both = [r for r in all_results
                    if r['methods'].get(m1,{}).get('status') == 'optimal'
                    and r['methods'].get(m2,{}).get('status') == 'optimal']
            if len(both) < 3:
                continue
            m1f = sum(1 for r in both if r['methods'][m1]['time'] < r['methods'][m2]['time'])
            ratios = [r['methods'][m2]['time']/max(r['methods'][m1]['time'],1e-6) for r in both]
            print(f"  {m1} vs {m2}: {len(both)} common, "
                  f"{m1} faster {m1f}/{len(both)}, geomean ratio={np.exp(np.mean(np.log(np.clip(ratios,1e-6,1e6)))):.2f}x")

    # Save
    out_dir = os.path.join(script_dir, 'bobilib_results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'results.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
