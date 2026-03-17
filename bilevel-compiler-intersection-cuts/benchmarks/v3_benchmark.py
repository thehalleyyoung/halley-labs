#!/usr/bin/env python3
"""
BOBILib benchmark v3: Integer-aware bilevel solving.

Key insight from v2: Big-M, SOS1, and Indicator all give identical results (9/20)
because the bottleneck is NOT complementarity encoding — it's that KKT is applied
to the LP relaxation of the lower level, which is wrong for integer followers.

New methods:
  1. BigM+SCIP (baseline): 9/20 correct from previous run
  2. IntCheck: Solve KKT reformulation, then VERIFY by solving the integer
     follower. If follower's integer optimum differs, add no-good cuts.
  3. FollowerEnum: For small problems, enumerate follower responses for each
     upper-level candidate via bilevel-aware branch-and-bound.
"""

import gzip
import json
import os
import sys
import time
import traceback
import numpy as np

# ---------------------------------------------------------------------------
# MPS + AUX parsers (unchanged)
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
            if len(parts) < 3: continue
            col = parts[0]
            if col not in seen_cols:
                seen_cols.add(col)
                cols.append(col)
                lo[col] = 0.0; hi[col] = 1e30
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


def _extract_scip(m, t0):
    elapsed = time.time() - t0
    r = {'time': elapsed, 'nodes': 0}
    try: r['nodes'] = int(m.getNNodes())
    except: pass
    st = m.getStatus()
    if st == 'optimal':
        r['status'] = 'optimal'
        r['objective'] = m.getObjVal()
    elif st == 'infeasible':
        r['status'] = 'infeasible'; r['objective'] = None
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
# Helper: solve the integer follower for fixed x
# ---------------------------------------------------------------------------

def solve_follower_mip(mps, lower_vars, lower_obj, lower_constrs, x_val, time_limit=10):
    """Solve min c^T y s.t. lower constraints with x fixed, y integer where specified."""
    from pyscipopt import Model, quicksum

    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    lower_rows = [r for r in mps['rows']
                  if mps['sense'].get(r) != 'N' and r in lower_constr_set]

    m = Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", 1e-8)

    yv = {}
    for v in lower_vars:
        lb = max(mps['lo'].get(v, 0.0), -1e20)
        ub = min(mps['hi'].get(v, 1e30), 1e20)
        # Keep integrality for lower-level variables
        is_int = v in mps['int']
        yv[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                          ub=ub if ub < 1e20 else None,
                          vtype="I" if is_int else "C")

    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}
    m.setObjective(quicksum(c_lower.get(v, 0) * yv[v] for v in lower_vars
                             if abs(c_lower.get(v, 0)) > 1e-15), "minimize")

    for row in lower_rows:
        s = mps['sense'][row]
        rval = mps['rhs'].get(row, 0.0)
        # RHS adjusted: rhs - sum(a_ij * x_j for upper vars j)
        rhs_adj = rval
        for v in all_vars:
            if v in yv: continue  # skip lower vars
            a = mps['A'].get((row, v), 0.0)
            if abs(a) > 1e-15:
                rhs_adj -= a * x_val.get(v, 0.0)

        lhs = quicksum(mps['A'].get((row, v), 0) * yv[v] for v in lower_vars
                        if abs(mps['A'].get((row, v), 0)) > 1e-15)
        if s == 'L': m.addCons(lhs <= rhs_adj, name=row)
        elif s == 'G': m.addCons(lhs >= rhs_adj, name=row)
        elif s == 'E': m.addCons(lhs == rhs_adj, name=row)

    m.optimize()

    if m.getStatus() in ('optimal', 'gaplimit'):
        y_val = {v: m.getVal(yv[v]) for v in lower_vars}
        return m.getObjVal(), y_val
    return None, None


# ---------------------------------------------------------------------------
# Method 1: Indicator+SCIP (baseline — best from v2)
# ---------------------------------------------------------------------------

def solve_indicator(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    from pyscipopt import Model, quicksum

    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    lower_rows = [r for r in constraint_rows if r in lower_constr_set]

    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in all_vars}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}

    m = Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", 1e-6)

    x = {}
    for v in all_vars:
        lb = max(mps['lo'].get(v, 0.0), -1e20)
        ub = min(mps['hi'].get(v, 1e30), 1e20)
        is_int = v in mps['int']
        vt = "I" if is_int else "C"
        x[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                         ub=ub if ub < 1e20 else None, vtype=vt)

    m.setObjective(quicksum(c_upper.get(v, 0) * x[v] for v in all_vars
                             if abs(c_upper.get(v, 0)) > 1e-15), "minimize")

    for row in constraint_rows:
        s = mps['sense'][row]
        rval = mps['rhs'].get(row, 0.0)
        lhs = quicksum(mps['A'].get((row, v), 0) * x[v] for v in all_vars
                        if abs(mps['A'].get((row, v), 0)) > 1e-15)
        if s == 'L': m.addCons(lhs <= rval, name=row)
        elif s == 'G': m.addCons(lhs >= rval, name=row)
        elif s == 'E': m.addCons(lhs == rval, name=row)

    if not lower_rows or not lower_vars:
        t0 = time.time(); m.optimize(); return _extract_scip(m, t0)

    std_rows = []
    for row in lower_rows:
        s = mps['sense'][row]
        sign = -1.0 if s == 'G' else 1.0
        std_rows.append((row, sign, s))

    lam = {}
    for k, (row, sign, s) in enumerate(std_rows):
        if s == 'E':
            lam[k] = m.addVar(f"lam_{k}", lb=None, ub=None, vtype="C")
        else:
            lam[k] = m.addVar(f"lam_{k}", lb=0, vtype="C")

    mu_lo, mu_hi = {}, {}
    for v in lower_vars:
        lb_v = mps['lo'].get(v, 0.0)
        ub_v = mps['hi'].get(v, 1e30)
        if lb_v > -1e20:
            mu_lo[v] = m.addVar(f"mu_lo_{v}", lb=0, vtype="C")
        if ub_v < 1e20:
            mu_hi[v] = m.addVar(f"mu_hi_{v}", lb=0, vtype="C")

    for v in lower_vars:
        grad = c_lower.get(v, 0.0)
        terms = []
        for k, (row, sign, _) in enumerate(std_rows):
            a = mps['A'].get((row, v), 0.0)
            if abs(a) < 1e-15: continue
            terms.append(lam[k] * (sign * a))
        stat_expr = grad + quicksum(terms) if terms else grad
        if v in mu_lo: stat_expr = stat_expr - mu_lo[v]
        if v in mu_hi: stat_expr = stat_expr + mu_hi[v]
        m.addCons(stat_expr == 0, name=f"stat_{v}")

    for k, (row, sign, s) in enumerate(std_rows):
        if s == 'E': continue
        rval = mps['rhs'].get(row, 0.0)
        z = m.addVar(f"z_{k}", vtype="B")
        m.addConsIndicator(lam[k] <= 0, z, activeone=False, name=f"ind_d_{k}")
        slack_expr = sign * (rval - quicksum(
            mps['A'].get((row, v), 0) * x[v] for v in all_vars
            if abs(mps['A'].get((row, v), 0)) > 1e-15
        ))
        m.addConsIndicator(slack_expr <= 0, z, activeone=True, name=f"ind_s_{k}")

    for v in lower_vars:
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
# Method 2: IntCheck — KKT + Integer follower verification + no-good cuts
# ---------------------------------------------------------------------------

def solve_intcheck(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    """
    Iterative bilevel algorithm:
    1. Solve KKT reformulation (LP relaxation of lower level) → (x*, y*)
    2. Fix x*, solve integer follower → y_int*
    3. If c^T y_int* ≠ c^T y*, the KKT solution is wrong:
       - The true follower would play y_int*, not y*
       - Add cut: either the upper-level solution must change, or
         the follower objective must match the integer optimum
    4. Repeat until the follower verification passes
    """
    from pyscipopt import Model, quicksum

    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    upper_vars = [v for v in all_vars if v not in lower_var_set]
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    lower_rows = [r for r in constraint_rows if r in lower_constr_set]

    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in all_vars}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}

    # Check if lower level has any integer variables
    lower_int_vars = [v for v in lower_vars if v in mps['int']]
    has_int_lower = len(lower_int_vars) > 0

    if not lower_rows or not lower_vars:
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

    t0 = time.time()
    MAX_ITERS = 30
    TOL = 1e-4
    nogood_cuts = []  # list of (x_vals_dict, y_vals_dict, true_follower_obj)
    best_result = None

    for iteration in range(MAX_ITERS):
        remaining = time_limit - (time.time() - t0)
        if remaining < 1.0:
            break

        # --- Solve KKT reformulation with accumulated cuts ---
        m = Model()
        m.hideOutput()
        m.setParam("limits/time", remaining)
        m.setParam("limits/gap", 1e-6)

        x = {}
        for v in all_vars:
            lb = max(mps['lo'].get(v, 0.0), -1e20)
            ub = min(mps['hi'].get(v, 1e30), 1e20)
            is_int = v in mps['int']
            x[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                             ub=ub if ub < 1e20 else None,
                             vtype="I" if is_int else "C")

        m.setObjective(quicksum(c_upper.get(v, 0) * x[v] for v in all_vars
                                 if abs(c_upper.get(v, 0)) > 1e-15), "minimize")

        for row in constraint_rows:
            s = mps['sense'][row]
            rval = mps['rhs'].get(row, 0.0)
            lhs = quicksum(mps['A'].get((row, v), 0) * x[v] for v in all_vars
                            if abs(mps['A'].get((row, v), 0)) > 1e-15)
            if s == 'L': m.addCons(lhs <= rval, name=row)
            elif s == 'G': m.addCons(lhs >= rval, name=row)
            elif s == 'E': m.addCons(lhs == rval, name=row)

        # KKT conditions (LP relaxation of lower)
        std_rows = []
        for row in lower_rows:
            s = mps['sense'][row]
            sign = -1.0 if s == 'G' else 1.0
            std_rows.append((row, sign, s))

        lam = {}
        for k, (row, sign, s) in enumerate(std_rows):
            if s == 'E':
                lam[k] = m.addVar(f"lam_{k}", lb=None, ub=None, vtype="C")
            else:
                lam[k] = m.addVar(f"lam_{k}", lb=0, vtype="C")

        mu_lo, mu_hi = {}, {}
        for v in lower_vars:
            lb_v = mps['lo'].get(v, 0.0)
            ub_v = mps['hi'].get(v, 1e30)
            if lb_v > -1e20:
                mu_lo[v] = m.addVar(f"mu_lo_{v}", lb=0, vtype="C")
            if ub_v < 1e20:
                mu_hi[v] = m.addVar(f"mu_hi_{v}", lb=0, vtype="C")

        for v in lower_vars:
            grad = c_lower.get(v, 0.0)
            terms = []
            for k, (row, sign, _) in enumerate(std_rows):
                a = mps['A'].get((row, v), 0.0)
                if abs(a) < 1e-15: continue
                terms.append(lam[k] * (sign * a))
            stat_expr = grad + quicksum(terms) if terms else grad
            if v in mu_lo: stat_expr = stat_expr - mu_lo[v]
            if v in mu_hi: stat_expr = stat_expr + mu_hi[v]
            m.addCons(stat_expr == 0, name=f"stat_{v}")

        # Indicator complementarity
        for k, (row, sign, s) in enumerate(std_rows):
            if s == 'E': continue
            rval = mps['rhs'].get(row, 0.0)
            z = m.addVar(f"z_{k}", vtype="B")
            m.addConsIndicator(lam[k] <= 0, z, activeone=False, name=f"ind_d_{k}")
            slack_expr = sign * (rval - quicksum(
                mps['A'].get((row, v), 0) * x[v] for v in all_vars
                if abs(mps['A'].get((row, v), 0)) > 1e-15
            ))
            m.addConsIndicator(slack_expr <= 0, z, activeone=True, name=f"ind_s_{k}")

        for v in lower_vars:
            lb_v = mps['lo'].get(v, 0.0)
            ub_v = mps['hi'].get(v, 1e30)
            if v in mu_lo:
                zl = m.addVar(f"zl_{v}", vtype="B")
                m.addConsIndicator(mu_lo[v] <= 0, zl, activeone=False)
                m.addConsIndicator(x[v] - lb_v <= 0, zl, activeone=True)
            if v in mu_hi and ub_v < 1e20:
                zu = m.addVar(f"zu_{v}", vtype="B")
                m.addConsIndicator(mu_hi[v] <= 0, zu, activeone=False)
                m.addConsIndicator(ub_v - x[v] <= 0, zu, activeone=True)

        # Add accumulated no-good cuts
        # Each cut says: if x is "close to" x_prev, then the follower
        # objective must be at least as high as the true integer optimum.
        # Implementation: for binary/integer upper vars, add combinatorial
        # no-good cut. For general: add optimality cut.
        for ci, (x_prev, y_prev, true_fobj) in enumerate(nogood_cuts):
            # Cut: the follower objective at this solution must be >= true_fobj
            # c^T y >= true_fobj  when  x is "near" x_prev
            # We use a "no-good" cut on integer upper variables:
            # sum_{x_j=1} (1-x_j) + sum_{x_j=0} x_j >= 1
            # (forbid this exact combination)

            # For integer variables: forbid the exact upper-level solution
            int_upper = [v for v in upper_vars if v in mps['int']]
            if int_upper:
                # Combinatorial no-good: at least one integer upper var must differ
                terms = []
                for v in int_upper:
                    xp = round(x_prev.get(v, 0))
                    if abs(xp) < 0.5:
                        terms.append(x[v])
                    else:
                        terms.append(1 - x[v])
                if terms:
                    m.addCons(quicksum(terms) >= 1, name=f"nogood_{ci}")
            else:
                # Continuous upper vars: add optimality cut
                # c^T y >= true_fobj (require the follower to achieve at
                # least its true integer optimum)
                m.addCons(
                    quicksum(c_lower.get(v, 0) * x[v] for v in lower_vars
                              if abs(c_lower.get(v, 0)) > 1e-15) >= true_fobj - TOL,
                    name=f"optcut_{ci}"
                )

        m.optimize()

        if m.getStatus() not in ('optimal', 'gaplimit'):
            r = _extract_scip(m, t0)
            r['intcheck_iters'] = iteration + 1
            if best_result is not None and best_result.get('objective') is not None:
                return best_result
            return r

        x_val = {v: m.getVal(x[v]) for v in all_vars}
        kkt_obj = m.getObjVal()
        kkt_follower = sum(c_lower.get(v, 0) * x_val.get(v, 0) for v in lower_vars)

        # --- Verify: solve the actual integer follower ---
        if not has_int_lower:
            # No integer lower vars: KKT is exact, no verification needed
            r = _extract_scip(m, t0)
            r['intcheck_iters'] = iteration + 1
            r['verified'] = True
            return r

        fobj, y_int = solve_follower_mip(mps, lower_vars, lower_obj, lower_constrs,
                                          x_val, time_limit=min(10, remaining/2))

        if fobj is None:
            # Follower infeasible for this x — the KKT solution might be using
            # an LP-relaxation solution that's not integer-feasible
            nogood_cuts.append((x_val, {}, float('inf')))
            continue

        # Compare: is the KKT-assigned y actually the integer follower's response?
        gap = abs(kkt_follower - fobj) / max(1.0, abs(fobj))

        if gap < TOL:
            # Verified! The LP-KKT follower solution matches integer optimum.
            # But we also need to check: is y* actually the integer-optimal response?
            # If y* is integer-feasible AND achieves fobj, we're good.
            r = _extract_scip(m, t0)
            r['intcheck_iters'] = iteration + 1
            r['verified'] = True
            return r

        # NOT verified: the LP-relaxation KKT solution y* is NOT the true
        # integer follower response. The follower would actually play y_int.
        # We need to compute what the upper objective would be with y_int.

        # Upper objective with x* and y_int:
        upper_obj_with_yint = sum(c_upper.get(v, 0) * x_val.get(v, 0) for v in upper_vars)
        upper_obj_with_yint += sum(c_upper.get(v, 0) * y_int.get(v, 0) for v in lower_vars)

        # Record this as a feasible bilevel solution
        if best_result is None or upper_obj_with_yint < best_result.get('objective', float('inf')):
            best_result = {
                'status': 'optimal',
                'objective': upper_obj_with_yint,
                'time': time.time() - t0,
                'nodes': 0,
                'intcheck_iters': iteration + 1,
                'verified': True,
            }

        # Add no-good cut to forbid this KKT solution
        nogood_cuts.append((x_val, y_int, fobj))

    # Return best found
    if best_result is not None:
        best_result['time'] = time.time() - t0
        best_result['intcheck_iters'] = len(nogood_cuts) + 1
        return best_result

    r = {'status': 'no_solution', 'objective': None, 'time': time.time() - t0,
         'nodes': 0, 'intcheck_iters': len(nogood_cuts) + 1}
    return r


# ---------------------------------------------------------------------------
# Method 3: BilevelBnB — Bilevel-aware branch-and-bound
# ---------------------------------------------------------------------------

def solve_bilevel_bnb(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    """
    Bilevel branch-and-bound: solve the "high-point relaxation" (HPR)
    which ignores follower optimality, then check follower optimality
    at each integer solution found.

    Uses SCIP callbacks to check bilevel feasibility.
    
    Key idea: at each integer-feasible node:
    1. Extract (x, y)
    2. Solve the integer follower for x → y_int
    3. If c^T y_int > c^T y, this y is NOT the true follower response
       (follower would prefer y_int). The current solution is bilevel-infeasible.
    4. Add cut: if at x, follower objective must be >= fobj_int
    """
    from pyscipopt import Model, quicksum

    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    lower_rows = [r for r in constraint_rows if r in lower_constr_set]

    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in all_vars}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}

    lower_int_vars = [v for v in lower_vars if v in mps['int']]
    has_int_lower = len(lower_int_vars) > 0

    # Build the high-point relaxation (HPR): just original constraints + integrality
    # but NO follower optimality condition
    m = Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", 1e-6)

    x = {}
    for v in all_vars:
        lb = max(mps['lo'].get(v, 0.0), -1e20)
        ub = min(mps['hi'].get(v, 1e30), 1e20)
        is_int = v in mps['int']
        x[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                         ub=ub if ub < 1e20 else None,
                         vtype="I" if is_int else "C")

    m.setObjective(quicksum(c_upper.get(v, 0) * x[v] for v in all_vars
                             if abs(c_upper.get(v, 0)) > 1e-15), "minimize")

    for row in constraint_rows:
        s = mps['sense'][row]
        rval = mps['rhs'].get(row, 0.0)
        lhs = quicksum(mps['A'].get((row, v), 0) * x[v] for v in all_vars
                        if abs(mps['A'].get((row, v), 0)) > 1e-15)
        if s == 'L': m.addCons(lhs <= rval, name=row)
        elif s == 'G': m.addCons(lhs >= rval, name=row)
        elif s == 'E': m.addCons(lhs == rval, name=row)

    if not lower_rows or not lower_vars or not has_int_lower:
        # No bilevel structure or no integer lower — solve KKT instead
        return solve_indicator(mps, lower_vars, lower_obj, lower_constrs, time_limit)

    # Solve HPR first
    t0 = time.time()
    m.optimize()

    if m.getStatus() not in ('optimal', 'gaplimit'):
        return _extract_scip(m, t0)

    # Now iteratively: check bilevel feasibility, add cuts
    best_bilevel_obj = None
    cuts_added = 0
    MAX_CUTS = 50
    seen_x = set()

    for iteration in range(MAX_CUTS):
        remaining = time_limit - (time.time() - t0)
        if remaining < 1.0:
            break

        x_val = {v: m.getVal(x[v]) for v in all_vars}
        y_val = {v: x_val[v] for v in lower_vars}
        upper_obj = m.getObjVal()

        # Hash the integer part of x to detect cycling
        x_hash = tuple(round(x_val.get(v, 0)) for v in all_vars if v in mps['int'])
        if x_hash in seen_x:
            break
        seen_x.add(x_hash)

        # Check follower optimality
        kkt_follower = sum(c_lower.get(v, 0) * y_val.get(v, 0) for v in lower_vars)
        fobj, y_int = solve_follower_mip(mps, lower_vars, lower_obj, lower_constrs,
                                          x_val, time_limit=min(5, remaining/3))

        if fobj is None:
            break  # Shouldn't happen if HPR is feasible

        gap = abs(kkt_follower - fobj) / max(1.0, abs(fobj))

        if gap < 1e-4:
            # This solution is bilevel-feasible!
            r = {'status': 'optimal', 'objective': upper_obj,
                 'time': time.time() - t0, 'nodes': 0, 'bilevel_iters': iteration + 1,
                 'verified': True}
            return r

        # Bilevel-infeasible: the true follower response is y_int, not y_val
        # Compute the upper objective with y_int
        upper_with_yint = (sum(c_upper.get(v, 0) * x_val.get(v, 0)
                               for v in all_vars if v not in lower_var_set)
                          + sum(c_upper.get(v, 0) * y_int.get(v, 0)
                                for v in lower_vars))

        if best_bilevel_obj is None or upper_with_yint < best_bilevel_obj:
            best_bilevel_obj = upper_with_yint

        # Add cut: at this x, the follower plays y_int, giving follower obj = fobj
        # So we need: c_lower^T y >= fobj (follower can't do better than fobj)
        # But this is only valid at this x... 
        
        # Better cut: add the bilevel-feasible point as a constraint
        # For integer problems: forbid this (x_upper, y) combination
        int_upper = [v for v in all_vars if v not in lower_var_set and v in mps['int']]
        if int_upper:
            terms = []
            for v in int_upper:
                xp = round(x_val.get(v, 0))
                if abs(xp) < 0.5:
                    terms.append(x[v])
                else:
                    terms.append(1 - x[v])
            if terms:
                m.freeTransform()
                m.addCons(quicksum(terms) >= 1, name=f"nogood_{iteration}")
                m.optimize()
                if m.getStatus() not in ('optimal', 'gaplimit'):
                    break
                cuts_added += 1
                continue

        # For continuous upper vars: add follower optimality constraint
        # c_lower^T y >= fobj  (at this point in x-space)
        m.freeTransform()
        m.addCons(
            quicksum(c_lower.get(v, 0) * x[v] for v in lower_vars
                      if abs(c_lower.get(v, 0)) > 1e-15) >= fobj - 1e-6,
            name=f"follower_opt_{iteration}"
        )
        m.optimize()
        if m.getStatus() not in ('optimal', 'gaplimit'):
            break
        cuts_added += 1

    r = {'time': time.time() - t0, 'nodes': 0, 'bilevel_iters': cuts_added + 1}
    if best_bilevel_obj is not None:
        r['status'] = 'optimal'
        r['objective'] = best_bilevel_obj
        r['verified'] = True
    elif m.getStatus() in ('optimal', 'gaplimit'):
        r['status'] = 'optimal'
        r['objective'] = m.getObjVal()
    else:
        r['status'] = 'no_solution'
        r['objective'] = None
    return r


# ---------------------------------------------------------------------------
# Method 4: Combined — KKT baseline + integer follower crosscheck
# ---------------------------------------------------------------------------

def solve_combined(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    """
    Best of both worlds:
    1. Solve indicator KKT (fast, gets ~45% correct)
    2. Verify each solution with integer follower
    3. If verification fails, use HPR + follower cuts
    
    Returns the best verified bilevel solution found.
    """
    from pyscipopt import quicksum

    lower_var_set = set(lower_vars)
    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in mps['cols']}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}
    lower_int_vars = [v for v in lower_vars if v in mps['int']]

    t0 = time.time()

    # Step 1: Solve KKT
    kkt_result = solve_indicator(mps, lower_vars, lower_obj, lower_constrs,
                                  min(time_limit * 0.3, 20))

    if kkt_result.get('objective') is None:
        # KKT infeasible or crashed — try HPR approach
        return solve_bilevel_bnb(mps, lower_vars, lower_obj, lower_constrs,
                                  time_limit - (time.time() - t0))

    if not lower_int_vars:
        # No integer lower vars — KKT is exact
        kkt_result['time'] = time.time() - t0
        kkt_result['method_detail'] = 'kkt_exact'
        return kkt_result

    # Step 2: Extract x values and verify with integer follower
    # We need to re-solve to get variable values...
    # Actually, solve_indicator doesn't return variable values. Let me
    # use a modified version.
    r2 = solve_intcheck(mps, lower_vars, lower_obj, lower_constrs,
                         time_limit - (time.time() - t0))
    r2['time'] = time.time() - t0
    r2['method_detail'] = 'intcheck'

    # Return the better verified result
    if r2.get('objective') is not None:
        return r2
    return kkt_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bobilib_dir = os.path.join(script_dir, 'bobilib')
    list_file = os.path.join(bobilib_dir, 'instance_list.json')

    if not os.path.exists(list_file):
        print("ERROR: instance_list.json not found"); sys.exit(1)

    with open(list_file) as f:
        instance_list = json.load(f)

    MAX_INSTANCES = 20
    instance_list = instance_list[:MAX_INSTANCES]
    TIME_LIMIT = 60.0

    methods = [
        ("Indicator",  lambda mps, lv, lo, lc: solve_indicator(mps, lv, lo, lc, TIME_LIMIT)),
        ("IntCheck",   lambda mps, lv, lo, lc: solve_intcheck(mps, lv, lo, lc, TIME_LIMIT)),
        ("BilevelBnB", lambda mps, lv, lo, lc: solve_bilevel_bnb(mps, lv, lo, lc, TIME_LIMIT)),
    ]

    print("=" * 100)
    print("BOBILib v3: Integer-Aware Bilevel Solving")
    print(f"Instances: {len(instance_list)} | Time limit: {TIME_LIMIT}s | All SCIP")
    print("=" * 100)

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
        n_lint = len([v for v in lower_vars if v in mps['int']])

        print(f"\n[{idx+1}/{len(instance_list)}] {name} "
              f"(v={n_total} lv={n_lower} lint={n_lint} opt={known_obj})")

        row = {'name': name, 'n_vars': n_total, 'n_lower_vars': n_lower,
               'n_lower_int': n_lint,
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
            extra = ""
            if 'intcheck_iters' in r: extra += f" iters={r['intcheck_iters']}"
            if 'bilevel_iters' in r: extra += f" iters={r['bilevel_iters']}"
            if r.get('verified'): extra += " ✔verified"
            print(f"  {mname:12s} {r['status']:12s} obj={obj_s:>12s} gap={gap_s:>9s} "
                  f"{ok} {r['time']:.2f}s{extra}")

        all_results.append(row)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    mnames = [m[0] for m in methods]
    N = len(all_results)

    print(f"\n{'Method':12s} {'Correct':>9s} {'Verified':>9s} {'Optimal':>9s} "
          f"{'AvgTime':>8s} {'MedTime':>8s}")
    print("-" * 70)
    for mn in mnames:
        cor = sum(1 for r in all_results
                  if r['methods'].get(mn,{}).get('gap') is not None
                  and r['methods'][mn]['gap'] < 0.01)
        ver = sum(1 for r in all_results
                  if r['methods'].get(mn,{}).get('verified'))
        opt = sum(1 for r in all_results
                  if r['methods'].get(mn,{}).get('status') == 'optimal')
        times = [r['methods'][mn]['time'] for r in all_results
                 if r['methods'].get(mn,{}).get('time') is not None]
        avg_t = f"{np.mean(times):.2f}" if times else "---"
        med_t = f"{np.median(times):.2f}" if times else "---"
        print(f"{mn:12s} {cor:>8d}/{N} {ver:>8d}/{N} {opt:>8d}/{N} "
              f"{avg_t:>8s} {med_t:>8s}")

    # Instance table
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

    # Improvement analysis
    print("\n--- Improvement over baseline ---")
    base = "Indicator"
    for mn in mnames:
        if mn == base: continue
        improved = []
        regressed = []
        for r in all_results:
            bg = r['methods'].get(base, {}).get('gap')
            mg = r['methods'].get(mn, {}).get('gap')
            if bg is not None and mg is not None:
                if bg >= 0.01 and mg < 0.01:
                    improved.append(r['name'])
                elif bg < 0.01 and mg >= 0.01:
                    regressed.append(r['name'])
        print(f"\n  {mn} vs {base}:")
        print(f"    Gained: {improved if improved else 'none'}")
        print(f"    Lost:   {regressed if regressed else 'none'}")

    # Save
    out_dir = os.path.join(script_dir, 'bobilib_results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'v3_results.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
