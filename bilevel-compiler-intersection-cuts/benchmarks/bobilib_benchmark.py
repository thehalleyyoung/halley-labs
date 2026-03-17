#!/usr/bin/env python3
"""
Honest BOBILib benchmark: compare bilevel solving approaches on real instances.

Approaches:
  1. KKT reformulation → HiGHS  (open-source MIP solver)
  2. KKT reformulation → SCIP   (open-source MIP solver)
  3. Big-M reformulation → HiGHS
  4. Big-M reformulation → SCIP

All compared against known-optimal BOBILib solutions.
"""

import gzip
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# BOBILib parser
# ---------------------------------------------------------------------------

@dataclass
class BilevelInstance:
    name: str
    # Full MPS data
    obj_name: str = ""
    row_names: list = field(default_factory=list)
    row_senses: dict = field(default_factory=dict)   # row_name -> 'N','L','G','E'
    col_names: list = field(default_factory=list)
    A: dict = field(default_factory=dict)             # (row,col) -> coeff
    rhs: dict = field(default_factory=dict)           # row -> value
    bounds_lo: dict = field(default_factory=dict)     # col -> lower
    bounds_hi: dict = field(default_factory=dict)     # col -> upper
    integers: set = field(default_factory=set)        # set of integer col names
    # Bilevel structure from .aux
    lower_vars: list = field(default_factory=list)    # names
    lower_var_obj: dict = field(default_factory=dict) # var -> lower-level obj coeff
    lower_constrs: list = field(default_factory=list) # names
    # Known solution
    known_obj: Optional[float] = None
    difficulty: str = ""
    status: str = ""


def parse_mps(path: str) -> dict:
    """Parse a (possibly gzipped) fixed-format MPS file."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='replace') as f:
        lines = f.readlines()

    data = {
        'obj_name': '', 'rows': [], 'row_sense': {},
        'cols': [], 'A': {}, 'rhs': {},
        'lo': {}, 'hi': {}, 'integers': set(),
        'col_order': [],
    }
    seen_cols = set()
    section = None
    in_integer = False

    for raw_line in lines:
        line = raw_line.rstrip('\n')
        if not line or line.startswith('*'):
            continue

        # Section headers start in column 0 with a letter
        if line[0] != ' ':
            token = line.split()[0]
            if token in ('NAME', 'ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'ENDATA'):
                section = token
                continue

        if section == 'ROWS':
            parts = line.split()
            if len(parts) >= 2:
                sense, name = parts[0], parts[1]
                data['rows'].append(name)
                data['row_sense'][name] = sense
                if sense == 'N':
                    data['obj_name'] = name

        elif section == 'COLUMNS':
            # Check for integer markers
            if "'MARKER'" in line:
                if "'INTORG'" in line:
                    in_integer = True
                elif "'INTEND'" in line:
                    in_integer = False
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            col = parts[0]
            if col not in seen_cols:
                seen_cols.add(col)
                data['col_order'].append(col)
                data['lo'][col] = 0.0
                data['hi'][col] = 1e30
            if in_integer:
                data['integers'].add(col)
            # pairs of (row, value)
            i = 1
            while i + 1 < len(parts):
                row, val = parts[i], float(parts[i+1])
                data['A'][(row, col)] = val
                i += 2

        elif section == 'RHS':
            parts = line.split()
            if len(parts) < 3:
                continue
            i = 1
            while i + 1 < len(parts):
                row, val = parts[i], float(parts[i+1])
                data['rhs'][row] = val
                i += 2

        elif section == 'BOUNDS':
            parts = line.split()
            if len(parts) < 3:
                continue
            btype = parts[0]
            col = parts[2]
            val = float(parts[3]) if len(parts) > 3 else 0.0
            if btype == 'UP':
                data['hi'][col] = val
            elif btype == 'LO':
                data['lo'][col] = val
            elif btype == 'FX':
                data['lo'][col] = val
                data['hi'][col] = val
            elif btype == 'FR':
                data['lo'][col] = -1e30
                data['hi'][col] = 1e30
            elif btype == 'MI':
                data['lo'][col] = -1e30
            elif btype == 'PL':
                data['hi'][col] = 1e30
            elif btype == 'BV':
                data['lo'][col] = 0.0
                data['hi'][col] = 1.0
                data['integers'].add(col)

    data['cols'] = data['col_order']
    return data


def parse_aux(path: str) -> dict:
    """Parse a BOBILib .aux file."""
    with open(path) as f:
        lines = f.read().strip().split('\n')

    aux = {'lower_vars': [], 'lower_var_obj': {}, 'lower_constrs': [],
           'name': '', 'mps_file': ''}
    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == '@VARSBEGIN':
            section = 'vars'; continue
        elif line == '@VARSEND':
            section = None; continue
        elif line == '@CONSTRSBEGIN':
            section = 'constrs'; continue
        elif line == '@CONSTRSEND':
            section = None; continue
        elif line.startswith('@NUMVARS') or line.startswith('@NUMCONSTRS'):
            section = 'skip'; continue
        elif line.startswith('@NAME'):
            section = 'name'; continue
        elif line.startswith('@MPS'):
            section = 'mps'; continue

        if section == 'vars':
            parts = line.split()
            var_name = parts[0]
            obj_coeff = float(parts[1]) if len(parts) > 1 else 0.0
            aux['lower_vars'].append(var_name)
            aux['lower_var_obj'][var_name] = obj_coeff
        elif section == 'constrs':
            parts = line.split()
            aux['lower_constrs'].append(parts[0])
        elif section == 'name':
            aux['name'] = line
            section = None
        elif section == 'mps':
            aux['mps_file'] = line
            section = None
        elif section == 'skip':
            section = None  # just skip the number line

    return aux


def load_instance(aux_path: str, sol_dir: str = None) -> Optional[BilevelInstance]:
    """Load a BOBILib instance from .aux + .mps.gz files."""
    aux = parse_aux(aux_path)
    base_dir = os.path.dirname(aux_path)
    mps_name = aux['mps_file'] or os.path.basename(aux_path).replace('.aux', '.mps')
    mps_path = os.path.join(base_dir, mps_name + '.gz')
    if not os.path.exists(mps_path):
        mps_path = os.path.join(base_dir, mps_name)
    if not os.path.exists(mps_path):
        return None

    mps = parse_mps(mps_path)
    inst = BilevelInstance(name=aux['name'] or os.path.basename(aux_path).replace('.aux', ''))
    inst.obj_name = mps['obj_name']
    inst.row_names = mps['rows']
    inst.row_senses = mps['row_sense']
    inst.col_names = mps['cols']
    inst.A = mps['A']
    inst.rhs = mps['rhs']
    inst.bounds_lo = mps['lo']
    inst.bounds_hi = mps['hi']
    inst.integers = mps['integers']
    inst.lower_vars = aux['lower_vars']
    inst.lower_var_obj = aux['lower_var_obj']
    inst.lower_constrs = aux['lower_constrs']

    # Load known solution if available
    if sol_dir:
        sol_file = os.path.join(sol_dir, inst.name + '.res.json')
        if os.path.exists(sol_file):
            with open(sol_file) as f:
                sol = json.load(f)
            inst.known_obj = sol.get('objective_value')
            inst.difficulty = sol.get('difficulty', '')
            inst.status = sol.get('status', '')

    return inst


# ---------------------------------------------------------------------------
# Approach 1 & 3: KKT / Big-M reformulation → HiGHS
# ---------------------------------------------------------------------------

def solve_kkt_highs(inst: BilevelInstance, time_limit: float = 60.0) -> dict:
    """KKT reformulation solved via HiGHS."""
    import highspy
    return _solve_reformulation(inst, 'highs', 'kkt', time_limit)


def solve_bigm_highs(inst: BilevelInstance, time_limit: float = 60.0, big_m: float = 1e4) -> dict:
    """Big-M complementarity reformulation solved via HiGHS."""
    import highspy
    return _solve_reformulation(inst, 'highs', 'bigm', time_limit, big_m)


def solve_kkt_scip(inst: BilevelInstance, time_limit: float = 60.0) -> dict:
    """KKT reformulation solved via SCIP."""
    return _solve_reformulation(inst, 'scip', 'kkt', time_limit)


def solve_bigm_scip(inst: BilevelInstance, time_limit: float = 60.0, big_m: float = 1e4) -> dict:
    """Big-M complementarity reformulation solved via SCIP."""
    return _solve_reformulation(inst, 'scip', 'bigm', time_limit, big_m)


def _solve_reformulation(inst: BilevelInstance, solver: str, method: str,
                         time_limit: float, big_m: float = 1e4) -> dict:
    """
    Build a single-level reformulation of the bilevel problem and solve.

    The bilevel problem structure:
      Upper level: min c_upper^T x  (over upper vars)
        s.t. upper constraints
      Lower level: min c_lower^T y  (over lower vars, parametric in x)
        s.t. lower constraints

    KKT reformulation: replace the lower-level optimality with:
      - lower primal feasibility
      - dual feasibility (stationarity)
      - complementary slackness: lambda_i * slack_i = 0

    Big-M linearization: lambda_i * slack_i = 0 becomes
      lambda_i <= M * z_i, slack_i <= M * (1 - z_i), z_i binary
    """
    result = {'method': f'{method}_{solver}', 'status': 'error', 'objective': None,
              'time': 0.0, 'gap': None, 'nodes': 0}

    # Identify upper vs lower variables and constraints
    lower_var_set = set(inst.lower_vars)
    lower_constr_set = set(inst.lower_constrs)

    all_vars = inst.col_names
    upper_vars = [v for v in all_vars if v not in lower_var_set]

    # Separate constraints
    constraint_rows = [r for r in inst.row_names if inst.row_senses.get(r, 'N') != 'N']
    upper_constrs = [r for r in constraint_rows if r not in lower_constr_set]
    lower_constrs = [r for r in constraint_rows if r in lower_constr_set]

    if not lower_constrs or not inst.lower_vars:
        result['status'] = 'skip_trivial'
        return result

    var_idx = {v: i for i, v in enumerate(all_vars)}
    n = len(all_vars)

    # Upper-level objective coefficients (from MPS objective row)
    c_upper = np.zeros(n)
    for v in all_vars:
        c_upper[var_idx[v]] = inst.A.get((inst.obj_name, v), 0.0)

    # Lower-level objective from .aux
    c_lower = np.zeros(n)
    for v in inst.lower_vars:
        c_lower[var_idx[v]] = inst.lower_var_obj.get(v, 0.0)

    try:
        if solver == 'highs':
            return _build_and_solve_highs(inst, method, time_limit, big_m,
                                          all_vars, var_idx, upper_vars,
                                          upper_constrs, lower_constrs,
                                          c_upper, c_lower, lower_var_set, result)
        else:
            return _build_and_solve_scip(inst, method, time_limit, big_m,
                                         all_vars, var_idx, upper_vars,
                                         upper_constrs, lower_constrs,
                                         c_upper, c_lower, lower_var_set, result)
    except Exception as e:
        result['status'] = f'error: {str(e)[:80]}'
        return result


def _build_and_solve_highs(inst, method, time_limit, big_m,
                            all_vars, var_idx, upper_vars,
                            upper_constrs, lower_constrs,
                            c_upper, c_lower, lower_var_set, result):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("mip_rel_gap", 1e-6)

    n = len(all_vars)
    # Add original variables
    for v in all_vars:
        lo = inst.bounds_lo.get(v, 0.0)
        hi = inst.bounds_hi.get(v, 1e30)
        idx = h.addVar(lo, hi)

    # Set upper-level objective
    for i, v in enumerate(all_vars):
        h.changeColCost(i, c_upper[i])

    # Mark integer variables (upper level only for now, lower level handled by KKT)
    for v in all_vars:
        if v in inst.integers and v not in lower_var_set:
            h.changeColIntegrality(var_idx[v], highspy._core.HighsVarType.kInteger)

    # Add upper-level constraints
    for row in upper_constrs:
        sense = inst.row_senses[row]
        rhs_val = inst.rhs.get(row, 0.0)
        coeffs = [(var_idx[c], inst.A[(row, c)]) for c in all_vars if (row, c) in inst.A]
        if not coeffs:
            continue
        indices, vals = zip(*coeffs)
        if sense == 'L':
            h.addRow(-highspy.kHighsInf, rhs_val, len(indices), list(indices), list(vals))
        elif sense == 'G':
            h.addRow(rhs_val, highspy.kHighsInf, len(indices), list(indices), list(vals))
        elif sense == 'E':
            h.addRow(rhs_val, rhs_val, len(indices), list(indices), list(vals))

    # Add lower-level primal feasibility constraints
    lower_slack_info = []  # (row, slack_var_idx, sense)
    for row in lower_constrs:
        sense = inst.row_senses[row]
        rhs_val = inst.rhs.get(row, 0.0)
        coeffs = [(var_idx[c], inst.A[(row, c)]) for c in all_vars if (row, c) in inst.A]
        if not coeffs:
            continue
        indices, vals = zip(*coeffs)
        if sense == 'L':
            h.addRow(-highspy.kHighsInf, rhs_val, len(indices), list(indices), list(vals))
        elif sense == 'G':
            h.addRow(rhs_val, highspy.kHighsInf, len(indices), list(indices), list(vals))
        elif sense == 'E':
            h.addRow(rhs_val, rhs_val, len(indices), list(indices), list(vals))
        lower_slack_info.append((row, sense, rhs_val))

    if method == 'kkt':
        # Add dual variables (one per lower constraint)
        dual_start = h.getNumCol()
        for i, (row, sense, _) in enumerate(lower_slack_info):
            if sense == 'L':
                h.addVar(0.0, 1e30)  # lambda >= 0
            elif sense == 'G':
                h.addVar(-1e30, 0.0)  # lambda <= 0 (or flip sign)
            elif sense == 'E':
                h.addVar(-1e30, 1e30)  # free

        # Stationarity: for each lower-level variable y_j:
        #   d(c_lower)/dy_j + sum_i lambda_i * A[i,j] = 0
        # (with bounds: if y_j at lower bound, >= 0; at upper, <= 0)
        # For simplicity, add as equality (interior point)
        for v in inst.lower_vars:
            j = var_idx[v]
            # Coefficient of lower obj
            obj_coeff = c_lower[j]
            # Gather lambda contributions
            indices = [j]  # dummy for building
            vals_list = []

            row_indices = []
            row_vals = []
            for k, (row, sense, _) in enumerate(lower_slack_info):
                a_val = inst.A.get((row, v), 0.0)
                if abs(a_val) > 1e-15:
                    row_indices.append(dual_start + k)
                    if sense == 'G':
                        row_vals.append(-a_val)  # converted to <= internally
                    else:
                        row_vals.append(a_val)

            if not row_indices:
                # Stationarity: obj_coeff = 0
                if abs(obj_coeff) > 1e-10:
                    # Infeasible stationarity — skip, solver will handle
                    pass
                continue

            # Add constraint: obj_coeff + sum(lambda_i * A[i,j]) = 0
            # Only dual vars in this constraint
            lo_bound = inst.bounds_lo.get(v, 0.0)
            hi_bound = inst.bounds_hi.get(v, 1e30)

            if lo_bound > -1e20 and hi_bound < 1e20:
                # Bounded: stationarity with slack
                h.addRow(-obj_coeff, -obj_coeff, len(row_indices), row_indices, row_vals)
            elif lo_bound > -1e20:
                # Lower bounded only: gradient >= 0
                h.addRow(-obj_coeff, highspy.kHighsInf, len(row_indices), row_indices, row_vals)
            elif hi_bound < 1e20:
                h.addRow(-highspy.kHighsInf, -obj_coeff, len(row_indices), row_indices, row_vals)
            else:
                h.addRow(-obj_coeff, -obj_coeff, len(row_indices), row_indices, row_vals)

        # Complementary slackness via strong duality
        # primal_obj = dual_obj  →  c_lower^T y = sum_i lambda_i * b_i
        # This avoids binary variables entirely
        sd_primal_indices = []
        sd_primal_vals = []
        for v in inst.lower_vars:
            j = var_idx[v]
            if abs(c_lower[j]) > 1e-15:
                sd_primal_indices.append(j)
                sd_primal_vals.append(c_lower[j])

        sd_dual_indices = []
        sd_dual_vals = []
        for k, (row, sense, rhs_val) in enumerate(lower_slack_info):
            if abs(rhs_val) > 1e-15:
                sd_dual_indices.append(dual_start + k)
                if sense == 'G':
                    sd_dual_vals.append(-rhs_val)
                else:
                    sd_dual_vals.append(rhs_val)
            # Also need bound contributions
            for v in inst.lower_vars:
                a_val = inst.A.get((row, v), 0.0)
                # Actually strong duality: sum b_i lambda_i includes bounds

        # Strong duality: c_lower^T y - sum lambda_i b_i = 0
        # Combine into one row
        all_sd_idx = sd_primal_indices + sd_dual_indices
        all_sd_val = sd_primal_vals + [-v for v in sd_dual_vals]
        if all_sd_idx:
            h.addRow(0.0, 0.0, len(all_sd_idx), all_sd_idx, all_sd_val)

    elif method == 'bigm':
        # Big-M complementarity linearization
        dual_start = h.getNumCol()
        for i, (row, sense, _) in enumerate(lower_slack_info):
            if sense == 'L':
                h.addVar(0.0, big_m)
            elif sense == 'G':
                h.addVar(-big_m, 0.0)
            elif sense == 'E':
                h.addVar(-big_m, big_m)

        # Stationarity (same as KKT)
        for v in inst.lower_vars:
            j = var_idx[v]
            obj_coeff = c_lower[j]
            row_indices = []
            row_vals = []
            for k, (row, sense, _) in enumerate(lower_slack_info):
                a_val = inst.A.get((row, v), 0.0)
                if abs(a_val) > 1e-15:
                    row_indices.append(dual_start + k)
                    row_vals.append(a_val if sense != 'G' else -a_val)
            if row_indices:
                h.addRow(-obj_coeff, -obj_coeff, len(row_indices), row_indices, row_vals)

        # Big-M complementarity: for each lower constraint i:
        #   lambda_i <= M * z_i
        #   slack_i  <= M * (1 - z_i)
        z_start = h.getNumCol()
        num_ineq = sum(1 for _, s, _ in lower_slack_info if s in ('L', 'G'))
        for i in range(num_ineq):
            h.addVar(0.0, 1.0)
            h.changeColIntegrality(z_start + i, highspy._core.HighsVarType.kInteger)

        z_idx = 0
        for k, (row, sense, rhs_val) in enumerate(lower_slack_info):
            if sense == 'E':
                continue
            # lambda_k - M*z_k <= 0
            h.addRow(-highspy.kHighsInf, 0.0, 2,
                     [dual_start + k, z_start + z_idx],
                     [1.0, -big_m] if sense == 'L' else [-1.0, -big_m])

            # slack_k <= M*(1-z_k): a^T x - rhs + M*z_k <= M
            coeffs_row = []
            vals_row = []
            for c in all_vars:
                a_val = inst.A.get((row, c), 0.0)
                if abs(a_val) > 1e-15:
                    coeffs_row.append(var_idx[c])
                    if sense == 'L':
                        vals_row.append(-a_val)  # slack = rhs - a^T x
                    else:
                        vals_row.append(a_val)
            coeffs_row.append(z_start + z_idx)
            vals_row.append(big_m)
            rhs_adj = big_m + (rhs_val if sense == 'L' else -rhs_val)
            h.addRow(-highspy.kHighsInf, rhs_adj, len(coeffs_row), coeffs_row, vals_row)

            z_idx += 1

    # Also mark lower-level integer variables
    for v in inst.lower_vars:
        if v in inst.integers:
            h.changeColIntegrality(var_idx[v], highspy._core.HighsVarType.kInteger)

    t0 = time.time()
    h.run()
    elapsed = time.time() - t0

    status = h.getInfoValue("primal_solution_status")[1]
    model_status = h.getModelStatus()

    result['time'] = elapsed
    result['nodes'] = int(h.getInfoValue("mip_node_count")[1]) if hasattr(h, 'getInfoValue') else 0

    if model_status == highspy.HighsModelStatus.kOptimal:
        result['status'] = 'optimal'
        result['objective'] = h.getInfoValue("objective_function_value")[1]
    elif model_status == highspy.HighsModelStatus.kObjectiveBound:
        result['status'] = 'feasible'
        result['objective'] = h.getInfoValue("objective_function_value")[1]
    elif model_status == highspy.HighsModelStatus.kInfeasible:
        result['status'] = 'infeasible'
    elif model_status == highspy.HighsModelStatus.kTimeLimit:
        result['status'] = 'time_limit'
        try:
            result['objective'] = h.getInfoValue("objective_function_value")[1]
        except:
            pass
    else:
        result['status'] = f'other: {model_status}'

    if result['objective'] is not None and inst.known_obj is not None and inst.known_obj != 0:
        result['gap'] = abs(result['objective'] - inst.known_obj) / max(1.0, abs(inst.known_obj))
    elif result['objective'] is not None and inst.known_obj is not None:
        result['gap'] = abs(result['objective'] - inst.known_obj)

    return result


def _build_and_solve_scip(inst, method, time_limit, big_m,
                           all_vars, var_idx, upper_vars,
                           upper_constrs, lower_constrs,
                           c_upper, c_lower, lower_var_set, result):
    from pyscipopt import Model, quicksum

    m = Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    m.setParam("limits/gap", 1e-6)

    n = len(all_vars)
    x = {}
    for v in all_vars:
        lo = inst.bounds_lo.get(v, 0.0)
        hi = inst.bounds_hi.get(v, 1e30)
        if lo < -1e20:
            lo = None
        if hi > 1e20:
            hi = None
        vtype = "I" if v in inst.integers else "C"
        x[v] = m.addVar(name=v, lb=lo, ub=hi, vtype=vtype)

    # Upper-level objective
    m.setObjective(quicksum(c_upper[var_idx[v]] * x[v] for v in all_vars), "minimize")

    # Upper-level constraints
    for row in upper_constrs:
        sense = inst.row_senses[row]
        rhs_val = inst.rhs.get(row, 0.0)
        lhs = quicksum(inst.A.get((row, v), 0.0) * x[v] for v in all_vars
                        if abs(inst.A.get((row, v), 0.0)) > 1e-15)
        if sense == 'L':
            m.addCons(lhs <= rhs_val, name=f"upper_{row}")
        elif sense == 'G':
            m.addCons(lhs >= rhs_val, name=f"upper_{row}")
        elif sense == 'E':
            m.addCons(lhs == rhs_val, name=f"upper_{row}")

    # Lower-level primal feasibility
    lower_slack_info = []
    for row in lower_constrs:
        sense = inst.row_senses[row]
        rhs_val = inst.rhs.get(row, 0.0)
        lhs = quicksum(inst.A.get((row, v), 0.0) * x[v] for v in all_vars
                        if abs(inst.A.get((row, v), 0.0)) > 1e-15)
        if sense == 'L':
            m.addCons(lhs <= rhs_val, name=f"lower_{row}")
        elif sense == 'G':
            m.addCons(lhs >= rhs_val, name=f"lower_{row}")
        elif sense == 'E':
            m.addCons(lhs == rhs_val, name=f"lower_{row}")
        lower_slack_info.append((row, sense, rhs_val))

    if method == 'kkt':
        # Dual variables
        lam = {}
        for k, (row, sense, _) in enumerate(lower_slack_info):
            if sense == 'L':
                lam[k] = m.addVar(name=f"lam_{k}", lb=0.0, ub=None, vtype="C")
            elif sense == 'G':
                lam[k] = m.addVar(name=f"lam_{k}", lb=None, ub=0.0, vtype="C")
            elif sense == 'E':
                lam[k] = m.addVar(name=f"lam_{k}", lb=None, ub=None, vtype="C")

        # Stationarity for each lower-level variable
        for v in inst.lower_vars:
            j = var_idx[v]
            obj_coeff = c_lower[j]
            dual_sum = quicksum(
                lam[k] * (inst.A.get((row, v), 0.0) * (-1.0 if sense == 'G' else 1.0))
                for k, (row, sense, _) in enumerate(lower_slack_info)
                if abs(inst.A.get((row, v), 0.0)) > 1e-15
            )
            lo_bound = inst.bounds_lo.get(v, 0.0)
            hi_bound = inst.bounds_hi.get(v, 1e30)
            if lo_bound > -1e20 and hi_bound < 1e20:
                m.addCons(obj_coeff + dual_sum == 0, name=f"stat_{v}")
            elif lo_bound > -1e20:
                m.addCons(obj_coeff + dual_sum >= 0, name=f"stat_{v}")
            elif hi_bound < 1e20:
                m.addCons(obj_coeff + dual_sum <= 0, name=f"stat_{v}")
            else:
                m.addCons(obj_coeff + dual_sum == 0, name=f"stat_{v}")

        # Strong duality
        primal_obj = quicksum(c_lower[var_idx[v]] * x[v] for v in inst.lower_vars
                               if abs(c_lower[var_idx[v]]) > 1e-15)
        dual_obj = quicksum(
            lam[k] * (rhs_val * (-1.0 if sense == 'G' else 1.0))
            for k, (row, sense, rhs_val) in enumerate(lower_slack_info)
            if abs(rhs_val) > 1e-15
        )
        m.addCons(primal_obj == dual_obj, name="strong_duality")

    elif method == 'bigm':
        # Dual variables
        lam = {}
        for k, (row, sense, _) in enumerate(lower_slack_info):
            if sense == 'L':
                lam[k] = m.addVar(name=f"lam_{k}", lb=0.0, ub=big_m, vtype="C")
            elif sense == 'G':
                lam[k] = m.addVar(name=f"lam_{k}", lb=-big_m, ub=0.0, vtype="C")
            elif sense == 'E':
                lam[k] = m.addVar(name=f"lam_{k}", lb=-big_m, ub=big_m, vtype="C")

        # Stationarity
        for v in inst.lower_vars:
            j = var_idx[v]
            obj_coeff = c_lower[j]
            dual_sum = quicksum(
                lam[k] * (inst.A.get((row, v), 0.0) * (-1.0 if sense == 'G' else 1.0))
                for k, (row, sense, _) in enumerate(lower_slack_info)
                if abs(inst.A.get((row, v), 0.0)) > 1e-15
            )
            m.addCons(obj_coeff + dual_sum == 0, name=f"stat_{v}")

        # Big-M complementarity
        z_vars = {}
        z_idx = 0
        for k, (row, sense, rhs_val) in enumerate(lower_slack_info):
            if sense == 'E':
                continue
            z_vars[k] = m.addVar(name=f"z_{k}", vtype="B")
            # lambda_k <= M * z_k (for L) or -lambda_k <= M * z_k (for G)
            if sense == 'L':
                m.addCons(lam[k] <= big_m * z_vars[k], name=f"comp_dual_{k}")
            else:
                m.addCons(-lam[k] <= big_m * z_vars[k], name=f"comp_dual_{k}")

            # slack_k <= M * (1 - z_k)
            slack = rhs_val - quicksum(
                inst.A.get((row, v), 0.0) * x[v] for v in all_vars
                if abs(inst.A.get((row, v), 0.0)) > 1e-15
            ) if sense == 'L' else quicksum(
                inst.A.get((row, v), 0.0) * x[v] for v in all_vars
                if abs(inst.A.get((row, v), 0.0)) > 1e-15
            ) - rhs_val

            m.addCons(slack <= big_m * (1 - z_vars[k]), name=f"comp_slack_{k}")
            z_idx += 1

    t0 = time.time()
    m.optimize()
    elapsed = time.time() - t0

    result['time'] = elapsed
    result['nodes'] = int(m.getNNodes())

    scip_status = m.getStatus()
    if scip_status == 'optimal':
        result['status'] = 'optimal'
        result['objective'] = m.getObjVal()
    elif scip_status == 'infeasible':
        result['status'] = 'infeasible'
    elif scip_status in ('timelimit', 'gaplimit'):
        result['status'] = 'time_limit'
        try:
            result['objective'] = m.getObjVal()
        except:
            pass
    else:
        result['status'] = f'other: {scip_status}'
        try:
            result['objective'] = m.getObjVal()
        except:
            pass

    if result['objective'] is not None and inst.known_obj is not None and inst.known_obj != 0:
        result['gap'] = abs(result['objective'] - inst.known_obj) / max(1.0, abs(inst.known_obj))
    elif result['objective'] is not None and inst.known_obj is not None:
        result['gap'] = abs(result['objective'] - inst.known_obj)

    return result


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def find_instances(base_dir: str, max_count: int = 200):
    """Find BOBILib instances, preferring those with known optimal solutions."""
    instances = []
    sol_dirs = {}

    # Map category paths to solution directories
    for aux_path in sorted(glob.glob(os.path.join(base_dir, 'general-bilevel/**/*.aux'), recursive=True)):
        # Determine solution directory
        rel = os.path.relpath(aux_path, os.path.join(base_dir, 'general-bilevel'))
        parts = rel.split(os.sep)
        if len(parts) >= 2:
            cat = parts[0]  # pure-integer or mixed-integer
            subcat = parts[1]  # denegre, miplib2010, etc.
            sol_dir = os.path.join(base_dir, f'general-bilevel-solutions/{cat}-solutions/{subcat}')
        else:
            sol_dir = None

        inst = load_instance(aux_path, sol_dir)
        if inst is None:
            continue

        # Compute size
        n_lower = len(inst.lower_vars)
        n_total = len(inst.col_names)
        instances.append((inst, n_lower, n_total))

    # Sort: prefer instances with known optima, then by size
    def sort_key(item):
        inst = item[0]
        has_opt = 1 if inst.known_obj is not None and inst.status == 'optimal' else 0
        return (-has_opt, item[1], item[2])

    instances.sort(key=sort_key)
    return [item[0] for item in instances[:max_count]]


def run_benchmark(base_dir: str, time_limit: float = 60.0, max_instances: int = 80):
    """Run the full benchmark comparison."""
    print("=" * 80)
    print("BOBILib Benchmark: Open-Source Bilevel Solver Comparison")
    print("=" * 80)
    print(f"Time limit per instance per method: {time_limit}s")
    print()

    print("Loading instances...")
    instances = find_instances(base_dir, max_instances)
    print(f"Loaded {len(instances)} instances")

    # Categorize
    with_opt = [i for i in instances if i.known_obj is not None and i.status == 'optimal']
    easy = [i for i in with_opt if i.difficulty == 'easy']
    hard = [i for i in with_opt if i.difficulty == 'hard']
    print(f"  With known optimal: {len(with_opt)} (easy: {len(easy)}, hard: {len(hard)})")
    print()

    methods = [
        ("KKT+HiGHS", solve_kkt_highs),
        ("KKT+SCIP", solve_kkt_scip),
        ("BigM+HiGHS", lambda inst, tl=time_limit: solve_bigm_highs(inst, tl)),
        ("BigM+SCIP", lambda inst, tl=time_limit: solve_bigm_scip(inst, tl)),
    ]

    all_results = []

    for idx, inst in enumerate(instances):
        n_lower = len(inst.lower_vars)
        n_total = len(inst.col_names)
        n_lower_c = len(inst.lower_constrs)
        print(f"\n[{idx+1}/{len(instances)}] {inst.name} "
              f"(vars={n_total}, lower_vars={n_lower}, lower_constrs={n_lower_c}, "
              f"diff={inst.difficulty}, known_obj={inst.known_obj})")

        inst_results = {'name': inst.name, 'n_vars': n_total,
                        'n_lower_vars': n_lower, 'n_lower_constrs': n_lower_c,
                        'difficulty': inst.difficulty, 'known_obj': inst.known_obj,
                        'methods': {}}

        for method_name, solver_fn in methods:
            try:
                r = solver_fn(inst, time_limit)
            except Exception as e:
                r = {'method': method_name, 'status': f'crash: {str(e)[:60]}',
                     'objective': None, 'time': 0, 'gap': None, 'nodes': 0}

            inst_results['methods'][method_name] = r

            status_str = r['status'][:12]
            obj_str = f"{r['objective']:.4f}" if r['objective'] is not None else "None"
            gap_str = f"{r['gap']:.2e}" if r.get('gap') is not None else "N/A"
            print(f"  {method_name:15s}: status={status_str:12s} obj={obj_str:>15s} "
                  f"gap={gap_str:>10s} time={r['time']:.2f}s nodes={r.get('nodes',0)}")

        all_results.append(inst_results)

    # Summary
    print_summary(all_results, methods)
    return all_results


def print_summary(all_results, methods):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    method_names = [m[0] for m in methods]

    # Count successes
    for mname in method_names:
        optimal = sum(1 for r in all_results
                      if r['methods'].get(mname, {}).get('status') == 'optimal')
        feasible = sum(1 for r in all_results
                       if r['methods'].get(mname, {}).get('objective') is not None)
        correct = sum(1 for r in all_results
                      if r['methods'].get(mname, {}).get('gap') is not None
                      and r['methods'][mname]['gap'] < 0.01)
        times = [r['methods'][mname]['time'] for r in all_results
                 if r['methods'].get(mname, {}).get('status') == 'optimal']
        avg_time = np.mean(times) if times else float('nan')
        med_time = np.median(times) if times else float('nan')

        print(f"\n{mname}:")
        print(f"  Optimal:  {optimal}/{len(all_results)}")
        print(f"  Feasible: {feasible}/{len(all_results)}")
        print(f"  Correct (gap<1%): {correct}/{len(all_results)}")
        print(f"  Avg time (optimal): {avg_time:.3f}s")
        print(f"  Med time (optimal): {med_time:.3f}s")

    # Head-to-head comparison
    print("\n--- Head-to-Head (on instances both solved optimally) ---")
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i+1:]:
            both_opt = [r for r in all_results
                        if r['methods'].get(m1, {}).get('status') == 'optimal'
                        and r['methods'].get(m2, {}).get('status') == 'optimal']
            if not both_opt:
                continue
            m1_faster = sum(1 for r in both_opt
                            if r['methods'][m1]['time'] < r['methods'][m2]['time'])
            m2_faster = len(both_opt) - m1_faster
            m1_times = [r['methods'][m1]['time'] for r in both_opt]
            m2_times = [r['methods'][m2]['time'] for r in both_opt]
            ratio = np.mean([t2/max(t1, 1e-6) for t1, t2 in zip(m1_times, m2_times)])
            print(f"  {m1} vs {m2}: {len(both_opt)} common, "
                  f"{m1} faster {m1_faster}x, {m2} faster {m2_faster}x, "
                  f"avg time ratio {ratio:.2f}")

    # By difficulty
    for diff in ['easy', 'hard']:
        subset = [r for r in all_results if r['difficulty'] == diff]
        if not subset:
            continue
        print(f"\n--- {diff.upper()} instances ({len(subset)}) ---")
        for mname in method_names:
            opt = sum(1 for r in subset
                      if r['methods'].get(mname, {}).get('status') == 'optimal')
            correct = sum(1 for r in subset
                          if r['methods'].get(mname, {}).get('gap') is not None
                          and r['methods'][mname]['gap'] < 0.01)
            print(f"  {mname:15s}: optimal={opt}/{len(subset)}, correct={correct}/{len(subset)}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bobilib_dir = os.path.join(base_dir, 'bobilib')

    if not os.path.exists(os.path.join(bobilib_dir, 'general-bilevel')):
        print("ERROR: BOBILib data not found. Download from bobilib.org first.")
        sys.exit(1)

    results = run_benchmark(bobilib_dir, time_limit=60.0, max_instances=80)

    # Save results
    out_dir = os.path.join(base_dir, 'bobilib_results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'benchmark_results.json')

    # Convert for JSON serialization
    for r in results:
        for mname, mdata in r['methods'].items():
            for k, v in mdata.items():
                if isinstance(v, (np.floating, np.integer)):
                    mdata[k] = float(v)

    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")
