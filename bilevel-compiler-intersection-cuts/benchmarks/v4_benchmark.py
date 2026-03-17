#!/usr/bin/env python3
"""
BOBILib v4: Fixed combined approach.

Key findings from v3:
- Indicator (KKT) gets 9/20: correct on bmilplib instances, wrong on moore/linderoth/knapsack/p0033
- BilevelBnB gets 7/20: correct on moore/linderoth/knapsack/p0033, wrong on bmilplib
- BilevelBnB had a bug: recorded (x*, y_int) without checking feasibility of (x*, y_int)

Fix: full feasibility check when replacing y with follower response.
Combine: run both, take the best FEASIBLE + VERIFIED solution.
"""

import gzip, json, os, sys, time, traceback
import numpy as np

# ---------------------------------------------------------------------------
# Parsers (same as v3)
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
        if not line or line[0] == '*': continue
        if line[0] != ' ':
            tok = line.split()[0]
            if tok in ('NAME','ROWS','COLUMNS','RHS','RANGES','BOUNDS','ENDATA'):
                section = tok
            continue
        parts = line.split()
        if section == 'ROWS' and len(parts) >= 2:
            rows.append(parts[1]); row_sense[parts[1]] = parts[0]
            if parts[0] == 'N': obj_name = parts[1]
        elif section == 'COLUMNS':
            if "'MARKER'" in line:
                in_integer = "'INTORG'" in line; continue
            if len(parts) < 3: continue
            col = parts[0]
            if col not in seen_cols:
                seen_cols.add(col); cols.append(col); lo[col] = 0.0; hi[col] = 1e30
            if in_integer: integers.add(col)
            i = 1
            while i+1 < len(parts):
                A[(parts[i], col)] = float(parts[i+1]); i += 2
        elif section == 'RHS' and len(parts) >= 3:
            i = 1
            while i+1 < len(parts):
                rhs[parts[i]] = float(parts[i+1]); i += 2
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
            p = l.split(); lower_vars.append(p[0])
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
        r['status'] = 'optimal'; r['objective'] = m.getObjVal()
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
# Check feasibility of a full solution (x_val dict) against all constraints
# ---------------------------------------------------------------------------

def check_feasibility(mps, x_val, tol=1e-5):
    """Check if x_val satisfies all constraints and bounds."""
    for v in mps['cols']:
        val = x_val.get(v, 0.0)
        lb = mps['lo'].get(v, 0.0)
        ub = mps['hi'].get(v, 1e30)
        if val < lb - tol or val > ub + tol:
            return False
    for row in mps['rows']:
        s = mps['sense'].get(row)
        if s == 'N': continue
        lhs = sum(mps['A'].get((row, v), 0.0) * x_val.get(v, 0.0) for v in mps['cols'])
        rval = mps['rhs'].get(row, 0.0)
        if s == 'L' and lhs > rval + tol: return False
        if s == 'G' and lhs < rval - tol: return False
        if s == 'E' and abs(lhs - rval) > tol: return False
    return True

# ---------------------------------------------------------------------------
# Solve integer follower for fixed x
# ---------------------------------------------------------------------------

def solve_follower_mip(mps, lower_vars, lower_obj, lower_constrs, x_val, time_limit=10):
    from pyscipopt import Model, quicksum
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    lower_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N' and r in lower_constr_set]

    m = Model(); m.hideOutput()
    m.setParam("limits/time", time_limit); m.setParam("limits/gap", 1e-8)

    yv = {}
    for v in lower_vars:
        lb = max(mps['lo'].get(v, 0.0), -1e20)
        ub = min(mps['hi'].get(v, 1e30), 1e20)
        yv[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                          ub=ub if ub < 1e20 else None,
                          vtype="I" if v in mps['int'] else "C")

    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}
    m.setObjective(quicksum(c_lower.get(v, 0) * yv[v] for v in lower_vars
                             if abs(c_lower.get(v, 0)) > 1e-15), "minimize")

    for row in lower_rows:
        s = mps['sense'][row]; rval = mps['rhs'].get(row, 0.0)
        rhs_adj = rval
        for v in all_vars:
            if v in yv: continue
            a = mps['A'].get((row, v), 0.0)
            if abs(a) > 1e-15: rhs_adj -= a * x_val.get(v, 0.0)
        lhs = quicksum(mps['A'].get((row, v), 0) * yv[v] for v in lower_vars
                        if abs(mps['A'].get((row, v), 0)) > 1e-15)
        if s == 'L': m.addCons(lhs <= rhs_adj)
        elif s == 'G': m.addCons(lhs >= rhs_adj)
        elif s == 'E': m.addCons(lhs == rhs_adj)

    m.optimize()
    if m.getStatus() in ('optimal', 'gaplimit'):
        return m.getObjVal(), {v: m.getVal(yv[v]) for v in lower_vars}
    return None, None

# ---------------------------------------------------------------------------
# Method 1: Indicator KKT (baseline)
# ---------------------------------------------------------------------------

def solve_indicator(mps, lower_vars, lower_obj, lower_constrs, time_limit, return_vals=False):
    from pyscipopt import Model, quicksum
    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    lower_rows = [r for r in constraint_rows if r in lower_constr_set]
    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in all_vars}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}

    m = Model(); m.hideOutput()
    m.setParam("limits/time", time_limit); m.setParam("limits/gap", 1e-6)

    x = {}
    for v in all_vars:
        lb = max(mps['lo'].get(v, 0.0), -1e20)
        ub = min(mps['hi'].get(v, 1e30), 1e20)
        x[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                         ub=ub if ub < 1e20 else None,
                         vtype="I" if v in mps['int'] else "C")

    m.setObjective(quicksum(c_upper.get(v, 0) * x[v] for v in all_vars
                             if abs(c_upper.get(v, 0)) > 1e-15), "minimize")
    for row in constraint_rows:
        s = mps['sense'][row]; rval = mps['rhs'].get(row, 0.0)
        lhs = quicksum(mps['A'].get((row, v), 0) * x[v] for v in all_vars
                        if abs(mps['A'].get((row, v), 0)) > 1e-15)
        if s == 'L': m.addCons(lhs <= rval, name=row)
        elif s == 'G': m.addCons(lhs >= rval, name=row)
        elif s == 'E': m.addCons(lhs == rval, name=row)

    if not lower_rows or not lower_vars:
        t0 = time.time(); m.optimize()
        r = _extract_scip(m, t0)
        if return_vals and r.get('objective') is not None:
            r['x_val'] = {v: m.getVal(x[v]) for v in all_vars}
        return r

    std_rows = []
    for row in lower_rows:
        s = mps['sense'][row]; sign = -1.0 if s == 'G' else 1.0
        std_rows.append((row, sign, s))

    lam = {}
    for k, (row, sign, s) in enumerate(std_rows):
        lam[k] = m.addVar(f"lam_{k}", lb=None if s == 'E' else 0, ub=None, vtype="C")

    mu_lo, mu_hi = {}, {}
    for v in lower_vars:
        if mps['lo'].get(v, 0.0) > -1e20:
            mu_lo[v] = m.addVar(f"mu_lo_{v}", lb=0, vtype="C")
        if mps['hi'].get(v, 1e30) < 1e20:
            mu_hi[v] = m.addVar(f"mu_hi_{v}", lb=0, vtype="C")

    for v in lower_vars:
        grad = c_lower.get(v, 0.0); terms = []
        for k, (row, sign, _) in enumerate(std_rows):
            a = mps['A'].get((row, v), 0.0)
            if abs(a) < 1e-15: continue
            terms.append(lam[k] * (sign * a))
        stat_expr = grad + quicksum(terms) if terms else grad
        if v in mu_lo: stat_expr -= mu_lo[v]
        if v in mu_hi: stat_expr += mu_hi[v]
        m.addCons(stat_expr == 0, name=f"stat_{v}")

    for k, (row, sign, s) in enumerate(std_rows):
        if s == 'E': continue
        rval = mps['rhs'].get(row, 0.0)
        z = m.addVar(f"z_{k}", vtype="B")
        m.addConsIndicator(lam[k] <= 0, z, activeone=False)
        slack_expr = sign * (rval - quicksum(
            mps['A'].get((row, v), 0) * x[v] for v in all_vars
            if abs(mps['A'].get((row, v), 0)) > 1e-15))
        m.addConsIndicator(slack_expr <= 0, z, activeone=True)

    for v in lower_vars:
        lb_v = mps['lo'].get(v, 0.0); ub_v = mps['hi'].get(v, 1e30)
        if v in mu_lo:
            zl = m.addVar(f"zl_{v}", vtype="B")
            m.addConsIndicator(mu_lo[v] <= 0, zl, activeone=False)
            m.addConsIndicator(x[v] - lb_v <= 0, zl, activeone=True)
        if v in mu_hi and ub_v < 1e20:
            zu = m.addVar(f"zu_{v}", vtype="B")
            m.addConsIndicator(mu_hi[v] <= 0, zu, activeone=False)
            m.addConsIndicator(ub_v - x[v] <= 0, zu, activeone=True)

    t0 = time.time(); m.optimize()
    r = _extract_scip(m, t0)
    if return_vals and r.get('objective') is not None:
        r['x_val'] = {v: m.getVal(x[v]) for v in all_vars}
    return r


# ---------------------------------------------------------------------------
# Method 2: BilevelBnB with full feasibility checking (fixed)
# ---------------------------------------------------------------------------

def solve_bilevel_bnb(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    """HPR + follower verification + no-good cuts.
    KEY FIX: when recording (x, y_int), verify full constraint feasibility."""
    from pyscipopt import Model, quicksum

    lower_var_set = set(lower_vars)
    lower_constr_set = set(lower_constrs)
    all_vars = mps['cols']
    constraint_rows = [r for r in mps['rows'] if mps['sense'].get(r) != 'N']
    lower_rows = [r for r in constraint_rows if r in lower_constr_set]
    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in all_vars}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}
    lower_int = [v for v in lower_vars if v in mps['int']]

    if not lower_rows or not lower_vars or not lower_int:
        return solve_indicator(mps, lower_vars, lower_obj, lower_constrs, time_limit)

    t0 = time.time()
    best_obj = None
    MAX_CUTS = 80
    seen_x = set()

    # Build and solve HPR iteratively
    m = Model(); m.hideOutput()
    m.setParam("limits/time", time_limit); m.setParam("limits/gap", 1e-6)

    x = {}
    for v in all_vars:
        lb = max(mps['lo'].get(v, 0.0), -1e20)
        ub = min(mps['hi'].get(v, 1e30), 1e20)
        x[v] = m.addVar(name=v, lb=lb if lb > -1e20 else None,
                         ub=ub if ub < 1e20 else None,
                         vtype="I" if v in mps['int'] else "C")

    m.setObjective(quicksum(c_upper.get(v, 0) * x[v] for v in all_vars
                             if abs(c_upper.get(v, 0)) > 1e-15), "minimize")
    for row in constraint_rows:
        s = mps['sense'][row]; rval = mps['rhs'].get(row, 0.0)
        lhs = quicksum(mps['A'].get((row, v), 0) * x[v] for v in all_vars
                        if abs(mps['A'].get((row, v), 0)) > 1e-15)
        if s == 'L': m.addCons(lhs <= rval, name=row)
        elif s == 'G': m.addCons(lhs >= rval, name=row)
        elif s == 'E': m.addCons(lhs == rval, name=row)

    cuts_added = 0
    m.optimize()

    for iteration in range(MAX_CUTS):
        remaining = time_limit - (time.time() - t0)
        if remaining < 1.0: break
        if m.getStatus() not in ('optimal', 'gaplimit'): break

        x_val = {v: m.getVal(x[v]) for v in all_vars}

        # Detect cycling
        x_hash = tuple(round(x_val.get(v, 0) * 100) / 100 for v in all_vars if v in mps['int'])
        if x_hash in seen_x: break
        seen_x.add(x_hash)

        # Check: is y* the true integer follower response to x*?
        y_val = {v: x_val[v] for v in lower_vars}
        kkt_fobj = sum(c_lower.get(v, 0) * y_val.get(v, 0) for v in lower_vars)

        fobj, y_int = solve_follower_mip(mps, lower_vars, lower_obj, lower_constrs,
                                          x_val, time_limit=min(5, remaining/3))
        if fobj is None: break

        gap_f = abs(kkt_fobj - fobj) / max(1.0, abs(fobj))

        if gap_f < 1e-4:
            # y* IS the true follower response → bilevel-feasible
            obj_val = m.getObjVal()
            if best_obj is None or obj_val < best_obj:
                best_obj = obj_val
            r = {'status': 'optimal', 'objective': best_obj,
                 'time': time.time() - t0, 'nodes': 0,
                 'bilevel_iters': iteration + 1, 'verified': True}
            return r

        # y* is NOT the true follower response. The follower would play y_int.
        # Check if (x*, y_int) is feasible for ALL constraints.
        combined = dict(x_val)
        combined.update(y_int)
        if check_feasibility(mps, combined):
            # Valid bilevel solution!
            upper_obj = sum(c_upper.get(v, 0) * combined.get(v, 0) for v in all_vars)
            if best_obj is None or upper_obj < best_obj:
                best_obj = upper_obj

        # Add no-good cut: forbid this x combination
        int_vars = [v for v in all_vars if v in mps['int']]
        if int_vars:
            terms = []
            for v in int_vars:
                xp = round(x_val.get(v, 0))
                if abs(xp) < 0.5:
                    terms.append(x[v])
                else:
                    terms.append(1 - x[v])
            if terms:
                m.freeTransform()
                m.addCons(quicksum(terms) >= 1, name=f"nogood_{iteration}")
                m.optimize()
                cuts_added += 1
                continue

        # Continuous: add follower optimality bound
        m.freeTransform()
        m.addCons(
            quicksum(c_lower.get(v, 0) * x[v] for v in lower_vars
                      if abs(c_lower.get(v, 0)) > 1e-15) >= fobj - 1e-6,
            name=f"fopt_{iteration}"
        )
        m.optimize()
        cuts_added += 1

    r = {'time': time.time() - t0, 'nodes': 0, 'bilevel_iters': cuts_added + 1}
    if best_obj is not None:
        r['status'] = 'optimal'; r['objective'] = best_obj; r['verified'] = True
    else:
        r['status'] = 'no_solution'; r['objective'] = None
    return r


# ---------------------------------------------------------------------------
# Method 3: Combined — best of KKT and BilevelBnB
# ---------------------------------------------------------------------------

def solve_combined(mps, lower_vars, lower_obj, lower_constrs, time_limit):
    """Run KKT first (fast). Then run BilevelBnB. Return best feasible answer."""
    t0 = time.time()

    lower_var_set = set(lower_vars)
    c_upper = {v: mps['A'].get((mps['obj'], v), 0.0) for v in mps['cols']}
    c_lower = {v: lower_obj.get(v, 0.0) for v in lower_vars}
    lower_int = [v for v in lower_vars if v in mps['int']]

    candidates = []

    # Run KKT (fast)
    kkt_r = solve_indicator(mps, lower_vars, lower_obj, lower_constrs,
                             min(time_limit * 0.4, 25), return_vals=True)
    if kkt_r.get('objective') is not None and 'x_val' in kkt_r:
        x_val = kkt_r['x_val']
        # Verify with integer follower
        if lower_int:
            fobj, y_int = solve_follower_mip(mps, lower_vars, lower_obj, lower_constrs,
                                              x_val, time_limit=5)
            y_kkt = {v: x_val[v] for v in lower_vars}
            kkt_fobj = sum(c_lower.get(v, 0) * y_kkt.get(v, 0) for v in lower_vars)
            if fobj is not None:
                fgap = abs(kkt_fobj - fobj) / max(1.0, abs(fobj))
                if fgap < 1e-4:
                    # KKT solution is bilevel-feasible
                    candidates.append(('kkt_verified', kkt_r['objective']))
                else:
                    # KKT y is wrong; try (x, y_int)
                    combined = dict(x_val)
                    combined.update(y_int)
                    if check_feasibility(mps, combined):
                        obj_with_yint = sum(c_upper.get(v, 0) * combined.get(v, 0)
                                            for v in mps['cols'])
                        candidates.append(('kkt_corrected', obj_with_yint))
        else:
            # No integer lower vars — KKT is exact
            candidates.append(('kkt_exact', kkt_r['objective']))

    # Run BilevelBnB (uses remaining time)
    remaining = time_limit - (time.time() - t0)
    if remaining > 2.0:
        bnb_r = solve_bilevel_bnb(mps, lower_vars, lower_obj, lower_constrs, remaining)
        if bnb_r.get('objective') is not None and bnb_r.get('verified'):
            candidates.append(('bnb', bnb_r['objective']))

    # Return best
    if candidates:
        best = min(candidates, key=lambda c: c[1])
        return {'status': 'optimal', 'objective': best[1],
                'time': time.time() - t0, 'nodes': 0,
                'method_detail': best[0], 'verified': True}

    if kkt_r.get('objective') is not None:
        return {'status': 'optimal', 'objective': kkt_r['objective'],
                'time': time.time() - t0, 'nodes': 0,
                'method_detail': 'kkt_unverified'}

    return {'status': kkt_r.get('status', 'no_solution'), 'objective': None,
            'time': time.time() - t0, 'nodes': 0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    list_file = os.path.join(script_dir, 'bobilib', 'instance_list.json')
    if not os.path.exists(list_file):
        print("ERROR: instance_list.json not found"); sys.exit(1)

    with open(list_file) as f:
        instance_list = json.load(f)

    MAX = 20
    instance_list = instance_list[:MAX]
    TL = 60.0

    methods = [
        ("Indicator",  lambda mps, lv, lo, lc: solve_indicator(mps, lv, lo, lc, TL)),
        ("BilevelBnB", lambda mps, lv, lo, lc: solve_bilevel_bnb(mps, lv, lo, lc, TL)),
        ("Combined",   lambda mps, lv, lo, lc: solve_combined(mps, lv, lo, lc, TL)),
    ]

    print("=" * 100)
    print("BOBILib v4: Fixed Integer-Aware Bilevel Solving")
    print(f"Instances: {len(instance_list)} | Time limit: {TL}s | All SCIP")
    print("=" * 100)

    all_results = []
    for idx, entry in enumerate(instance_list):
        name = entry['name']
        aux_path = entry['aux_path']
        known_obj = entry['obj']

        mps_gz = aux_path.replace('.aux', '.mps.gz')
        try:
            mps = parse_mps_gz(mps_gz)
            lower_vars, lower_obj, lower_constrs = parse_aux(aux_path)
        except Exception as e:
            print(f"\n[{idx+1}] {name}: PARSE ERROR: {e}"); continue

        n_lint = len([v for v in lower_vars if v in mps['int']])
        print(f"\n[{idx+1}/{len(instance_list)}] {name} "
              f"(v={len(mps['cols'])} lv={len(lower_vars)} lint={n_lint} opt={known_obj})")

        row = {'name': name, 'known_obj': known_obj, 'methods': {}}

        for mname, fn in methods:
            try:
                r = fn(mps, lower_vars, lower_obj, lower_constrs)
            except Exception as e:
                r = {'status': 'crash', 'objective': None, 'time': 0}
                if '--debug' in sys.argv: traceback.print_exc()

            obj = r.get('objective')
            gap = None
            if obj is not None and known_obj is not None:
                gap = abs(obj - known_obj) / max(1.0, abs(known_obj)) if abs(known_obj) > 1e-8 else abs(obj - known_obj)
            r['gap'] = gap
            row['methods'][mname] = r

            obj_s = f"{obj:.2f}" if obj is not None else "None"
            gap_s = f"{gap:.1e}" if gap is not None else "---"
            ok = "✓" if gap is not None and gap < 0.01 else ("✗" if gap is not None else "?")
            extra = ""
            for k in ('bilevel_iters', 'method_detail'):
                if k in r: extra += f" {k}={r[k]}"
            print(f"  {mname:12s} {r['status']:12s} obj={obj_s:>12s} gap={gap_s:>9s} "
                  f"{ok} {r['time']:.2f}s{extra}")

        all_results.append(row)

    # Summary
    print("\n" + "=" * 100)
    mnames = [m[0] for m in methods]
    N = len(all_results)

    print(f"\n{'Method':12s} {'Correct':>9s} {'AvgTime':>8s}")
    print("-" * 40)
    for mn in mnames:
        cor = sum(1 for r in all_results
                  if r['methods'].get(mn,{}).get('gap') is not None
                  and r['methods'][mn]['gap'] < 0.01)
        times = [r['methods'][mn]['time'] for r in all_results
                 if r['methods'].get(mn,{}).get('time') is not None]
        avg_t = f"{np.mean(times):.2f}" if times else "---"
        print(f"{mn:12s} {cor:>8d}/{N} {avg_t:>8s}")

    print(f"\n{'Instance':30s}", end="")
    for mn in mnames:
        print(f"  {mn:>11s}", end="")
    print(f"  {'Known':>10s}")
    print("-" * (30 + 13 * len(mnames) + 12))
    for r in all_results:
        print(f"{r['name']:30s}", end="")
        for mn in mnames:
            obj = r['methods'].get(mn, {}).get('objective')
            gap = r['methods'].get(mn, {}).get('gap')
            if obj is not None:
                ok = "✓" if gap is not None and gap < 0.01 else " "
                print(f"  {obj:>9.1f}{ok} ", end="")
            else:
                print(f"  {'---':>10s} ", end="")
        ko = r['known_obj']
        print(f"  {ko:>10.1f}" if ko is not None else "  ---")

    # Improvement analysis
    base = "Indicator"
    for mn in mnames:
        if mn == base: continue
        gained = [r['name'] for r in all_results
                  if (r['methods'].get(base,{}).get('gap') is None or r['methods'][base]['gap'] >= 0.01)
                  and r['methods'].get(mn,{}).get('gap') is not None and r['methods'][mn]['gap'] < 0.01]
        lost = [r['name'] for r in all_results
                if r['methods'].get(base,{}).get('gap') is not None and r['methods'][base]['gap'] < 0.01
                and (r['methods'].get(mn,{}).get('gap') is None or r['methods'][mn]['gap'] >= 0.01)]
        print(f"\n{mn} vs {base}: +{len(gained)} -{len(lost)}")
        if gained: print(f"  Gained: {gained}")
        if lost: print(f"  Lost:   {lost}")

    out_dir = os.path.join(script_dir, 'bobilib_results')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'v4_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/v4_results.json")


if __name__ == '__main__':
    main()
