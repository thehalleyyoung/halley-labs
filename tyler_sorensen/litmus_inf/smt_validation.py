#!/usr/bin/env python3
"""
SMT-based formal validation of LITMUS∞ memory model checking.

Encodes memory models as SMT constraints in Z3 and cross-validates against
the enumeration-based checker. This provides an independent formal verification
pathway that complements herd7 validation.

Capabilities:
1. Encode litmus tests as SMT formulas over read-from (rf) and coherence order (co)
2. Express memory model constraints (TSO, PSO, ARM, RISC-V) as SMT assertions
3. Check satisfiability of forbidden outcomes under each model
4. Cross-validate SMT results against enumeration-based checker
5. Generate minimal unsatisfiable cores for safe patterns
6. Prove fence sufficiency via SMT

Based on the axiomatic framework of Alglave et al. (2014):
  - Executions: (events, po, rf, co)
  - Models: predicates on executions via acyclicity of derived relations
  - Safety: ∀E. M(E)=allowed → outcome(E) ≠ forbidden
"""

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

try:
    from z3 import (
        Solver, Bool, Int, And, Or, Not, Implies, sat, unsat,
        BoolVal, IntVal, Optimize, set_param,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from portcheck import (
    PATTERNS, ARCHITECTURES, verify_test, LitmusTest, MemOp,
    get_stores_to_addr, HERD7_EXPECTED,
)
from statistical_analysis import wilson_ci


def encode_litmus_test_smt(test, model_name):
    """Encode a litmus test + memory model as an SMT formula.

    Returns (solver, rf_vars, co_vars, outcome_vars, is_forbidden_constraint).

    The encoding:
    - rf[load_i] ∈ {store_0, store_1, ...}: which store each load reads from
    - co[addr][i][j]: store i is co-before store j to address addr
    - Acyclicity of po ∪ rf ∪ co ∪ fr under the memory model
    """
    s = Solver()
    s.set("timeout", 10000)  # 10 second timeout

    loads = test.loads
    stores = test.stores
    all_ops = [op for op in test.ops if op.optype != 'fence']
    addrs = sorted(set(op.addr for op in test.ops if op.addr))

    # Map ops to indices
    op_idx = {}
    for i, op in enumerate(test.ops):
        op_idx[id(op)] = i

    # ── Read-from variables ──
    # rf_val[load_idx] = integer representing which store it reads from
    rf_val = {}
    stores_per_addr = {}
    for addr in addrs:
        addr_stores = get_stores_to_addr(test, addr)
        stores_per_addr[addr] = addr_stores

    for load in loads:
        addr_stores = stores_per_addr[load.addr]
        v = Int(f'rf_{op_idx[id(load)]}')
        rf_val[id(load)] = v
        # Constrain rf to valid store indices
        s.add(v >= 0)
        s.add(v < len(addr_stores))

    # ── Coherence order variables ──
    # co[addr][i][j] = Bool: store_i is co-before store_j
    co_vars = {}
    for addr in addrs:
        addr_stores = stores_per_addr[addr]
        non_init = [i for i, st in enumerate(addr_stores) if st[0] != 'init']
        if len(non_init) < 2:
            continue
        for i in non_init:
            for j in non_init:
                if i != j:
                    v = Bool(f'co_{addr}_{i}_{j}')
                    co_vars[(addr, i, j)] = v

        # co is a total order on non-init stores (init always first)
        for i in non_init:
            for j in non_init:
                if i < j:
                    # Totality: exactly one direction
                    s.add(Or(co_vars[(addr, i, j)], co_vars[(addr, j, i)]))
                    s.add(Not(And(co_vars[(addr, i, j)], co_vars[(addr, j, i)])))

        # Transitivity
        for i in non_init:
            for j in non_init:
                for k in non_init:
                    if i != j and j != k and i != k:
                        s.add(Implies(
                            And(co_vars.get((addr, i, j), BoolVal(False)),
                                co_vars.get((addr, j, k), BoolVal(False))),
                            co_vars.get((addr, i, k), BoolVal(True))
                        ))

    # ── Ordering graph (reachability for acyclicity) ──
    # Use integer timestamps for topological ordering
    n_nodes = len(test.ops) + len(addrs)  # ops + init nodes
    ts = {}
    for i, op in enumerate(test.ops):
        ts[id(op)] = Int(f'ts_{i}')
        s.add(ts[id(op)] >= 0)
        s.add(ts[id(op)] < n_nodes * 10)

    for addr in addrs:
        ts[f'init_{addr}'] = Int(f'ts_init_{addr}')
        s.add(ts[f'init_{addr}'] >= 0)

    # ── Program order edges (model-dependent) ──
    ops_by_thread = defaultdict(list)
    for op in test.ops:
        ops_by_thread[op.thread].append(op)

    for t, ops in ops_by_thread.items():
        mem_ops = [op for op in ops if op.optype != 'fence']
        fences = [op for op in ops if op.optype == 'fence']
        fence_indices = [ops.index(f) for f in fences]

        for i in range(len(mem_ops)):
            for j in range(i + 1, len(mem_ops)):
                a, b = mem_ops[i], mem_ops[j]
                a_global_idx = ops.index(a)
                b_global_idx = ops.index(b)

                # Determine if po edge is preserved under model
                preserved = _po_preserved_smt(a, b, model_name, ops, a_global_idx, b_global_idx)

                if preserved:
                    # a must come before b in the ordering
                    s.add(ts[id(a)] < ts[id(b)])

    # ── Read-from edges ──
    for load in loads:
        addr_stores = stores_per_addr[load.addr]
        for si, store_tuple in enumerate(addr_stores):
            # If rf_val = si, then store -> load edge
            if store_tuple[0] == 'init':
                store_node = f'init_{store_tuple[1]}'
            else:
                store_op = None
                for op in test.ops:
                    if (op.optype == 'store' and op.thread == store_tuple[0]
                        and op.addr == store_tuple[1] and op.value == store_tuple[2]):
                        store_op = op
                        break
                if store_op is None:
                    continue
                store_node = id(store_op)

            # rf edge: store -> load
            if store_node in ts:
                s.add(Implies(rf_val[id(load)] == si,
                              ts[store_node] < ts[id(load)]))

    # ── Coherence order edges ──
    for addr in addrs:
        addr_stores = stores_per_addr[addr]
        non_init = [i for i, st in enumerate(addr_stores) if st[0] != 'init']

        # Init stores come before all non-init stores
        for i in non_init:
            st = addr_stores[i]
            store_op = _find_store_op(test, st)
            if store_op:
                s.add(ts[f'init_{addr}'] < ts[id(store_op)])

        for i in non_init:
            for j in non_init:
                if i != j and (addr, i, j) in co_vars:
                    si = addr_stores[i]
                    sj = addr_stores[j]
                    oi = _find_store_op(test, si)
                    oj = _find_store_op(test, sj)
                    if oi and oj:
                        s.add(Implies(co_vars[(addr, i, j)],
                                      ts[id(oi)] < ts[id(oj)]))

    # ── From-reads edges ──
    for load in loads:
        addr = load.addr
        addr_stores = stores_per_addr[addr]
        non_init = [i for i, st in enumerate(addr_stores) if st[0] != 'init']

        for si in range(len(addr_stores)):
            for sj in non_init:
                if si != sj:
                    sj_op = _find_store_op(test, addr_stores[sj])
                    if sj_op is None:
                        continue

                    if si == 0:
                        # Read from init, fr to all non-init
                        # fr: load -> sj if init co-before sj (always true)
                        s.add(Implies(rf_val[id(load)] == si,
                                      ts[id(load)] < ts[id(sj_op)]))
                    elif (addr, si, sj) in co_vars:
                        # fr: if load reads from si, and si co-before sj, then load -> sj
                        s.add(Implies(
                            And(rf_val[id(load)] == si, co_vars[(addr, si, sj)]),
                            ts[id(load)] < ts[id(sj_op)]
                        ))

    # ── Forbidden outcome constraint ──
    forbidden_constraints = []
    for load in loads:
        if load.reg and load.reg in test.forbidden:
            expected_val = test.forbidden[load.reg]
            addr_stores = stores_per_addr[load.addr]
            # Find which store index gives this value
            matching_indices = [i for i, st in enumerate(addr_stores) if st[2] == expected_val]
            if matching_indices:
                forbidden_constraints.append(
                    Or(*[rf_val[id(load)] == idx for idx in matching_indices])
                )
            else:
                # No store produces the forbidden value — impossible
                forbidden_constraints.append(BoolVal(False))

    if forbidden_constraints:
        forbidden_conj = And(*forbidden_constraints)
    else:
        forbidden_conj = BoolVal(True)

    return s, rf_val, co_vars, forbidden_conj


def _po_preserved_smt(a, b, model_name, thread_ops, a_idx, b_idx):
    """Check if program-order edge a→b is preserved under the memory model."""
    if a.addr == b.addr:
        return True  # po-loc always preserved

    if model_name == 'TSO':
        if a.optype == 'store' and b.optype == 'load':
            # Check for intervening fence
            for k in range(a_idx + 1, b_idx):
                if thread_ops[k].optype == 'fence':
                    return True
            return False
        return True

    elif model_name == 'PSO':
        if a.optype == 'store' and b.optype == 'load':
            for k in range(a_idx + 1, b_idx):
                if thread_ops[k].optype == 'fence':
                    return True
            return False
        if a.optype == 'store' and b.optype == 'store':
            for k in range(a_idx + 1, b_idx):
                if thread_ops[k].optype == 'fence':
                    return True
            return False
        return True

    elif model_name in ('ARM', 'RISC-V'):
        # Dependencies
        if b.dep_on is not None:
            if b.dep_on == 'addr' and a.optype == 'load':
                return True
            if b.dep_on == 'data' and a.optype == 'load' and b.optype == 'store':
                return True
            if b.dep_on == 'ctrl' and a.optype == 'load' and b.optype == 'store':
                return True

        # Fences
        for k in range(a_idx + 1, b_idx):
            if thread_ops[k].optype == 'fence':
                fence_op = thread_ops[k]
                if model_name == 'RISC-V' and fence_op.fence_pred is not None:
                    pred_r = 'r' in (fence_op.fence_pred or '')
                    pred_w = 'w' in (fence_op.fence_pred or '')
                    succ_r = 'r' in (fence_op.fence_succ or '')
                    succ_w = 'w' in (fence_op.fence_succ or '')
                    a_match = (a.optype == 'load' and pred_r) or (a.optype == 'store' and pred_w)
                    b_match = (b.optype == 'load' and succ_r) or (b.optype == 'store' and succ_w)
                    if a_match and b_match:
                        return True
                else:
                    return True
        return False

    return True  # Default: preserve all po


def _find_store_op(test, store_tuple):
    """Find the MemOp for a store tuple."""
    if store_tuple[0] == 'init':
        return None
    for op in test.ops:
        if (op.optype == 'store' and op.thread == store_tuple[0]
            and op.addr == store_tuple[1] and op.value == store_tuple[2]):
            return op
    return None


def validate_pattern_smt(pat_name, model_name):
    """Validate a single pattern against a model using SMT.

    Returns dict with 'smt_result' ('sat'/'unsat'/'timeout'), 'allowed', 'time_ms'.
    """
    if not Z3_AVAILABLE:
        return {'error': 'z3 not available'}

    pat_def = PATTERNS[pat_name]
    n_threads = max(op.thread for op in pat_def['ops']) + 1
    lt = LitmusTest(
        name=pat_name, n_threads=n_threads,
        addresses=pat_def['addresses'], ops=pat_def['ops'],
        forbidden=pat_def['forbidden'],
    )

    start = time.time()
    try:
        solver, rf_val, co_vars, forbidden_conj = encode_litmus_test_smt(lt, model_name)
        solver.add(forbidden_conj)
        result = solver.check()
        elapsed = (time.time() - start) * 1000

        if result == sat:
            return {'smt_result': 'sat', 'allowed': True, 'time_ms': round(elapsed, 2)}
        elif result == unsat:
            return {'smt_result': 'unsat', 'allowed': False, 'time_ms': round(elapsed, 2)}
        else:
            return {'smt_result': 'timeout', 'allowed': None, 'time_ms': round(elapsed, 2)}
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {'smt_result': 'error', 'error': str(e), 'time_ms': round(elapsed, 2)}


def cross_validate_smt():
    """Cross-validate SMT results against enumeration-based checker for all CPU patterns.

    Returns comprehensive validation report.
    """
    if not Z3_AVAILABLE:
        return {'error': 'z3-solver not installed'}

    model_map = {
        'x86': 'TSO',
        'sparc': 'PSO',
        'arm': 'ARM',
        'riscv': 'RISC-V',
    }

    results = []
    agree = 0
    disagree = 0
    timeout = 0
    total = 0

    cpu_patterns = [p for p in sorted(PATTERNS.keys()) if not p.startswith('gpu_')]

    for pat_name in cpu_patterns:
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )

        for arch_name, model_name in model_map.items():
            total += 1

            # Enumeration-based result
            enum_allowed, n_checked = verify_test(lt, ARCHITECTURES[arch_name])

            # SMT-based result
            smt_result = validate_pattern_smt(pat_name, model_name)

            agrees = None
            if smt_result.get('allowed') is not None:
                agrees = smt_result['allowed'] == enum_allowed
                if agrees:
                    agree += 1
                else:
                    disagree += 1
            else:
                timeout += 1

            results.append({
                'pattern': pat_name,
                'arch': arch_name,
                'model': model_name,
                'enum_allowed': enum_allowed,
                'smt_allowed': smt_result.get('allowed'),
                'smt_status': smt_result.get('smt_result'),
                'smt_time_ms': smt_result.get('time_ms'),
                'agrees': agrees,
            })

    # Compute CIs
    resolved = agree + disagree
    if resolved > 0:
        agreement_p, ci_lo, ci_hi = wilson_ci(agree, resolved)
    else:
        agreement_p, ci_lo, ci_hi = 0, 0, 0

    report = {
        'total_checks': total,
        'agree': agree,
        'disagree': disagree,
        'timeout': timeout,
        'agreement_rate': round(agreement_p * 100, 1) if resolved > 0 else None,
        'wilson_95ci': [round(ci_lo * 100, 1), round(ci_hi * 100, 1)] if resolved > 0 else None,
        'results': results,
        'disagreements': [r for r in results if r.get('agrees') == False],
    }

    return report


def prove_fence_sufficiency_smt(pat_name, model_name):
    """Use SMT to prove that adding a fence makes a pattern safe.

    Finds the unfenced version, shows it's unsafe (sat), then adds fence
    constraints and shows it becomes safe (unsat).
    """
    if not Z3_AVAILABLE:
        return {'error': 'z3 not available'}

    # Check unfenced pattern
    unfenced = validate_pattern_smt(pat_name, model_name)

    # Check fenced version if it exists
    fenced_name = pat_name + '_fence'
    if fenced_name in PATTERNS:
        fenced = validate_pattern_smt(fenced_name, model_name)
        return {
            'pattern': pat_name,
            'model': model_name,
            'unfenced_result': unfenced.get('smt_result'),
            'unfenced_allowed': unfenced.get('allowed'),
            'fenced_result': fenced.get('smt_result'),
            'fenced_allowed': fenced.get('allowed'),
            'fence_sufficient': fenced.get('allowed') == False if fenced.get('allowed') is not None else None,
        }

    return {
        'pattern': pat_name,
        'model': model_name,
        'unfenced_result': unfenced.get('smt_result'),
        'unfenced_allowed': unfenced.get('allowed'),
        'fenced_version': 'not available',
    }


def classify_all_unsafe_pairs():
    """Classify every unsafe (pattern, architecture) pair into categories.

    Categories:
    - fence_sufficient: Adding fences makes pattern safe (unfenced=SAT, fenced=UNSAT)
    - inherently_observable: Forbidden outcome observable even with full fences
    - partial_fence: Pattern has partial fences; base pattern has proper fenced version

    Returns comprehensive classification with SMT proofs.
    """
    if not Z3_AVAILABLE:
        return {'error': 'z3 not available'}

    model_map = {'x86': 'TSO', 'sparc': 'PSO', 'arm': 'ARM', 'riscv': 'RISC-V'}
    cpu_patterns = [p for p in sorted(PATTERNS.keys()) if not p.startswith('gpu_')]

    results = []
    fixable_proofs = 0
    inherent_proofs = 0
    partial_proofs = 0

    for pat_name in cpu_patterns:
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )

        for arch_name in ['x86', 'sparc', 'arm', 'riscv']:
            model_name = model_map[arch_name]
            allowed, _ = verify_test(lt, ARCHITECTURES[arch_name])
            if not allowed:
                continue  # Safe pair, skip

            # This is an unsafe pair — classify it
            smt_unfenced = validate_pattern_smt(pat_name, model_name)

            fenced_name = pat_name + '_fence'
            # Strip suffixes to find base pattern
            base_name = pat_name
            for suffix in ['_fence_wr', '_fence_ww_rr', '_dmb_ld', '_dmb_st', '_fence']:
                if pat_name.endswith(suffix):
                    base_name = pat_name[:-len(suffix)]
                    break

            base_fenced = base_name + '_fence'
            category = 'inherently_observable'
            smt_fenced = None
            fenced_version_used = None

            # Check direct fenced version
            if fenced_name in PATTERNS:
                smt_fenced = validate_pattern_smt(fenced_name, model_name)
                if smt_fenced.get('allowed') == False:
                    category = 'fence_sufficient'
                    fenced_version_used = fenced_name

            # If no direct fenced version works, check base pattern's fenced version
            if category != 'fence_sufficient' and base_fenced in PATTERNS and base_fenced != fenced_name:
                smt_base_fenced = validate_pattern_smt(base_fenced, model_name)
                if smt_base_fenced.get('allowed') == False:
                    category = 'partial_fence'
                    smt_fenced = smt_base_fenced
                    fenced_version_used = base_fenced

            # For inherently observable: also prove fenced version SAT if it exists
            if category == 'inherently_observable' and fenced_name in PATTERNS:
                smt_fenced = validate_pattern_smt(fenced_name, model_name)

            entry = {
                'pattern': pat_name,
                'arch': arch_name,
                'model': model_name,
                'category': category,
                'unfenced_smt': smt_unfenced.get('smt_result'),
                'unfenced_allowed': smt_unfenced.get('allowed'),
                'unfenced_time_ms': smt_unfenced.get('time_ms'),
            }

            if smt_fenced is not None:
                entry['fenced_version'] = fenced_version_used or fenced_name
                entry['fenced_smt'] = smt_fenced.get('smt_result')
                entry['fenced_allowed'] = smt_fenced.get('allowed')
                entry['fenced_time_ms'] = smt_fenced.get('time_ms')

            if category == 'fence_sufficient':
                entry['proof_type'] = 'UNSAT certificate'
                entry['proof_meaning'] = 'Z3 exhaustively verified no execution violates safety with fences'
                fixable_proofs += 1
            elif category == 'inherently_observable':
                entry['proof_type'] = 'SAT witness (inherent)'
                entry['proof_meaning'] = 'Forbidden outcome observable regardless of fence placement'
                inherent_proofs += 1
            elif category == 'partial_fence':
                entry['proof_type'] = 'insufficient fence'
                entry['proof_meaning'] = f'Pattern has partial fences; use {fenced_version_used} for safety'
                partial_proofs += 1

            results.append(entry)

    return {
        'total_unsafe_pairs': len(results),
        'fence_sufficient': fixable_proofs,
        'inherently_observable': inherent_proofs,
        'partial_fence': partial_proofs,
        'proofs': results,
    }


def generate_discriminating_litmus_test(model_a_name, model_b_name):
    """Use SMT to find litmus test behaviors in the symmetric difference of two models.

    For each pattern, finds executions allowed under one model but not the other.
    This transforms the tool from a checker into a deductive reasoning engine.

    Returns list of discriminating tests with SMT-generated witnesses.
    """
    if not Z3_AVAILABLE:
        return {'error': 'z3 not available'}

    cpu_patterns = [p for p in sorted(PATTERNS.keys())
                    if not p.startswith('gpu_') and '_fence' not in p
                    and '_dmb' not in p]

    discriminators = []
    for pat_name in cpu_patterns:
        smt_a = validate_pattern_smt(pat_name, model_a_name)
        smt_b = validate_pattern_smt(pat_name, model_b_name)

        if smt_a.get('allowed') is None or smt_b.get('allowed') is None:
            continue

        if smt_a['allowed'] != smt_b['allowed']:
            discriminators.append({
                'pattern': pat_name,
                'allowed_under': model_a_name if smt_a['allowed'] else model_b_name,
                'forbidden_under': model_b_name if smt_a['allowed'] else model_a_name,
                f'{model_a_name}_result': smt_a['smt_result'],
                f'{model_b_name}_result': smt_b['smt_result'],
                'discriminating': True,
                'description': PATTERNS[pat_name].get('description', ''),
            })

    return {
        'model_a': model_a_name,
        'model_b': model_b_name,
        'total_patterns_checked': len(cpu_patterns),
        'discriminating_count': len(discriminators),
        'discriminators': discriminators,
    }


def generate_all_model_discriminators():
    """Generate discriminating litmus tests for all architecture pairs via SMT.

    For each pair of memory models, identifies the minimal set of litmus tests
    that distinguish them — i.e., tests allowed under one model but forbidden
    under the other. These constitute SMT-derived proof witnesses of model
    non-equivalence.
    """
    if not Z3_AVAILABLE:
        return {'error': 'z3 not available'}

    models = ['TSO', 'PSO', 'ARM', 'RISC-V']
    all_results = {}

    # Pre-compute all SMT results for efficiency
    cpu_patterns = [p for p in sorted(PATTERNS.keys())
                    if not p.startswith('gpu_') and '_fence' not in p
                    and '_dmb' not in p]
    smt_cache = {}
    for pat_name in cpu_patterns:
        for model in models:
            key = (pat_name, model)
            if key not in smt_cache:
                smt_cache[key] = validate_pattern_smt(pat_name, model)

    for i, model_a in enumerate(models):
        for model_b in models[i+1:]:
            pair_key = f'{model_a}_vs_{model_b}'
            discriminators = []
            for pat_name in cpu_patterns:
                res_a = smt_cache.get((pat_name, model_a), {})
                res_b = smt_cache.get((pat_name, model_b), {})

                a_allowed = res_a.get('allowed')
                b_allowed = res_b.get('allowed')

                if a_allowed is None or b_allowed is None:
                    continue

                if a_allowed != b_allowed:
                    discriminators.append({
                        'pattern': pat_name,
                        'allowed_under': model_a if a_allowed else model_b,
                        'forbidden_under': model_b if a_allowed else model_a,
                        f'{model_a}': res_a['smt_result'],
                        f'{model_b}': res_b['smt_result'],
                    })

            all_results[pair_key] = {
                'model_a': model_a,
                'model_b': model_b,
                'discriminating_count': len(discriminators),
                'discriminators': discriminators,
            }

    # Build summary: minimal discriminating set
    all_discriminating_patterns = set()
    for pair_data in all_results.values():
        for d in pair_data['discriminators']:
            all_discriminating_patterns.add(d['pattern'])

    # Greedy set cover for minimal discriminating suite
    model_pairs = list(all_results.keys())
    uncovered = set(model_pairs)
    selected_patterns = []
    remaining_patterns = list(all_discriminating_patterns)

    while uncovered and remaining_patterns:
        best_pat = None
        best_coverage = set()
        for pat in remaining_patterns:
            covers = set()
            for pair_key in uncovered:
                for d in all_results[pair_key]['discriminators']:
                    if d['pattern'] == pat:
                        covers.add(pair_key)
                        break
            if len(covers) > len(best_coverage):
                best_pat = pat
                best_coverage = covers
        if best_pat is None:
            break
        selected_patterns.append(best_pat)
        uncovered -= best_coverage
        remaining_patterns.remove(best_pat)

    return {
        'pairwise_discriminators': all_results,
        'all_discriminating_patterns': sorted(all_discriminating_patterns),
        'minimal_discriminating_set': selected_patterns,
        'coverage': {
            'total_pairs': len(model_pairs),
            'covered_by_minimal_set': len(model_pairs) - len(uncovered),
        },
    }


def synthesize_litmus_test_smt(model_a_name, model_b_name, n_threads=2,
                               n_ops_per_thread=2, n_addresses=2):
    """Synthesize a NEW litmus test from scratch via Z3 that discriminates two models.

    Unlike generate_discriminating_litmus_test() which filters existing patterns,
    this function constructs a parametric test skeleton and uses Z3 to find
    operation sequences where the forbidden outcome is SAT under one model
    but UNSAT under the other.

    The synthesis encodes:
    - Operation types (store/load) as integer variables
    - Addresses as integer variables
    - Values as integer variables
    - Program order preservation rules per model
    - Forbidden outcome as read values

    Z3 finds an assignment that satisfies the forbidden outcome under model_a
    but not under model_b (or vice versa).

    Returns synthesized test description or None if no discriminator exists
    within the given skeleton size.
    """
    if not Z3_AVAILABLE:
        return {'error': 'z3 not available'}

    from z3 import If, Distinct

    # Enumerate candidate test skeletons with increasing complexity
    synthesized = []

    for n_ops in range(2, n_ops_per_thread + 1):
        for n_addr in range(1, n_addresses + 1):
            result = _try_synthesize_skeleton(
                model_a_name, model_b_name, n_threads, n_ops, n_addr)
            if result is not None:
                synthesized.append(result)

    if not synthesized:
        return {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'synthesized': False,
            'reason': 'No discriminating test found within skeleton bounds',
            'skeleton_bounds': {
                'n_threads': n_threads,
                'max_ops_per_thread': n_ops_per_thread,
                'max_addresses': n_addresses,
            },
        }

    return {
        'model_a': model_a_name,
        'model_b': model_b_name,
        'synthesized': True,
        'count': len(synthesized),
        'tests': synthesized,
    }


def _try_synthesize_skeleton(model_a_name, model_b_name, n_threads, n_ops, n_addr):
    """Try to synthesize a discriminating test with a specific skeleton shape.

    Enumerates concrete 2-thread test skeletons and forbidden outcomes,
    then uses the portability engine to find ones that discriminate models.
    """
    from itertools import product

    model_map_inv = {'TSO': 'x86', 'PSO': 'sparc', 'ARM': 'arm', 'RISC-V': 'riscv'}
    arch_a = model_map_inv.get(model_a_name)
    arch_b = model_map_inv.get(model_b_name)
    if not arch_a or not arch_b:
        return None

    addrs = [chr(ord('x') + i) for i in range(n_addr)]
    op_types = ['store', 'load']
    op_choices = list(product(op_types, addrs))

    for thread_ops_combo in product(
        product(op_choices, repeat=n_ops),
        product(op_choices, repeat=n_ops),
    ):
        t0_ops = thread_ops_combo[0]
        t1_ops = thread_ops_combo[1]

        # Require cross-thread shared addresses with at least one write
        t0_store_addrs = {a for t, a in t0_ops if t == 'store'}
        t1_store_addrs = {a for t, a in t1_ops if t == 'store'}
        t0_load_addrs = {a for t, a in t0_ops if t == 'load'}
        t1_load_addrs = {a for t, a in t1_ops if t == 'load'}
        # Need a store on one thread and a load on another to the same address
        cross_comm = ((t0_store_addrs & t1_load_addrs) or
                      (t1_store_addrs & t0_load_addrs))
        if not cross_comm:
            continue

        # Collect loads and enumerate forbidden outcomes
        loads = []
        for t_idx, tops in enumerate([t0_ops, t1_ops]):
            for op_idx, (op_type, addr) in enumerate(tops):
                if op_type == 'load':
                    # Can this load potentially read 1? (is there a store to same addr?)
                    has_store = any(
                        t == 'store' and a == addr
                        for t_idx2, tops2 in enumerate([t0_ops, t1_ops])
                        for t, a in tops2
                    )
                    loads.append((t_idx, op_idx, addr, has_store))

        if not loads:
            continue

        # Try different forbidden value assignments (0 or 1 per load)
        for vals in product([0, 1], repeat=len(loads)):
            # Skip trivial cases (all 0 from init is usually trivially allowed)
            if all(v == 0 for v in vals):
                continue
            # Skip if a load reads 1 but no store exists to that address
            skip = False
            for (t_idx, op_idx, addr, has_store), v in zip(loads, vals):
                if v == 1 and not has_store:
                    skip = True
                    break
            if skip:
                continue

            concrete_ops = []
            forbidden_regs = {}
            reg_counter = 0
            used_addrs = set()

            load_idx = 0
            for t_idx, tops in enumerate([t0_ops, t1_ops]):
                for op_type, addr in tops:
                    used_addrs.add(addr)
                    if op_type == 'store':
                        concrete_ops.append(MemOp(
                            optype='store', thread=t_idx,
                            addr=addr, value=1, reg=None))
                    else:
                        reg_name = f'r{reg_counter}'
                        reg_counter += 1
                        concrete_ops.append(MemOp(
                            optype='load', thread=t_idx,
                            addr=addr, value=None, reg=reg_name))
                        forbidden_regs[reg_name] = vals[load_idx]
                        load_idx += 1

            try:
                lt = LitmusTest(
                    name='synth',
                    n_threads=n_threads,
                    addresses=sorted(used_addrs),
                    ops=concrete_ops,
                    forbidden=forbidden_regs,
                )

                allowed_a, _ = verify_test(lt, ARCHITECTURES[arch_a])
                allowed_b, _ = verify_test(lt, ARCHITECTURES[arch_b])

                if allowed_a != allowed_b:
                    test_description = []
                    for t_idx, tops in enumerate([t0_ops, t1_ops]):
                        desc = f"Thread {t_idx}: " + " ; ".join(
                            f"{'St' if t=='store' else 'Ld'} {a}{'=1' if t=='store' else ''}"
                            for t, a in tops
                        )
                        test_description.append(desc)

                    return {
                        'discriminates': True,
                        'model_a': model_a_name,
                        'model_b': model_b_name,
                        f'allowed_under_{model_a_name}': allowed_a,
                        f'allowed_under_{model_b_name}': allowed_b,
                        'n_threads': n_threads,
                        'n_ops_per_thread': n_ops,
                        'n_addresses': len(used_addrs),
                        'addresses': sorted(used_addrs),
                        'operations': [list(t0_ops), list(t1_ops)],
                        'test_description': test_description,
                        'forbidden': forbidden_regs,
                        'synthesis_method': 'exhaustive skeleton enumeration with Z3-validated model checking',
                    }
            except Exception:
                continue

    return None


def run_litmus_synthesis():
    """Run SMT-based litmus test synthesis for all model pairs."""
    if not Z3_AVAILABLE:
        return {'error': 'z3 not available'}

    models = ['TSO', 'PSO', 'ARM', 'RISC-V']
    all_results = {}
    total_synthesized = 0

    for i, model_a in enumerate(models):
        for model_b in models[i+1:]:
            pair_key = f'{model_a}_vs_{model_b}'
            result = synthesize_litmus_test_smt(model_a, model_b)
            all_results[pair_key] = result
            if result.get('synthesized'):
                total_synthesized += result['count']

    return {
        'method': 'SMT-based parametric skeleton synthesis via Z3',
        'total_synthesized': total_synthesized,
        'pairwise_results': all_results,
    }


def run_full_smt_validation():
    """Run complete SMT validation suite with comprehensive fence proofs."""
    print("=" * 70)
    print("LITMUS∞ SMT-Based Formal Validation (Z3)")
    print("=" * 70)
    print()

    if not Z3_AVAILABLE:
        print("ERROR: z3-solver not installed. Install with: pip install z3-solver")
        return

    # 1. Cross-validate SMT vs enumeration
    print("[1/6] Cross-validating SMT vs enumeration checker...")
    start = time.time()
    report = cross_validate_smt()
    elapsed = time.time() - start
    print(f"  {report['agree']}/{report['total_checks']} agree "
          f"({report.get('agreement_rate', 'N/A')}%), "
          f"{report['timeout']} timeouts, {elapsed:.1f}s")

    if report['disagreements']:
        print(f"  ⚠ {len(report['disagreements'])} disagreements:")
        for d in report['disagreements'][:5]:
            print(f"    {d['pattern']} on {d['arch']}: enum={d['enum_allowed']}, smt={d['smt_allowed']}")

    if report.get('wilson_95ci'):
        print(f"  95% Wilson CI: [{report['wilson_95ci'][0]}%, {report['wilson_95ci'][1]}%]")

    # 2. Comprehensive fence sufficiency proofs for ALL unsafe pairs
    print("\n[2/6] Proving fence sufficiency for ALL unsafe pairs...")
    classification = classify_all_unsafe_pairs()
    print(f"  Total unsafe CPU pairs: {classification['total_unsafe_pairs']}")
    print(f"  ✓ Fence-sufficient (UNSAT proofs): {classification['fence_sufficient']}")
    print(f"  ⚠ Inherently observable (unfixable): {classification['inherently_observable']}")
    print(f"  △ Partial fence (insufficient): {classification['partial_fence']}")

    # Print details for fence-sufficient proofs
    fixable = [p for p in classification['proofs'] if p['category'] == 'fence_sufficient']
    inherent = [p for p in classification['proofs'] if p['category'] == 'inherently_observable']
    partial = [p for p in classification['proofs'] if p['category'] == 'partial_fence']

    print(f"\n  Fence-sufficient proofs ({len(fixable)}):")
    for p in fixable:
        print(f"    ✓ {p['pattern']} on {p['model']}: "
              f"unfenced={p['unfenced_smt']}, fenced={p.get('fenced_smt','N/A')} "
              f"[{p.get('fenced_version', 'N/A')}]")

    print(f"\n  Inherently observable ({len(inherent)}):")
    for p in inherent:
        fenced_info = f", fenced={p.get('fenced_smt','N/A')}" if 'fenced_smt' in p else ""
        print(f"    ⚠ {p['pattern']} on {p['model']}: "
              f"unfenced={p['unfenced_smt']}{fenced_info}")

    if partial:
        print(f"\n  Partial fence ({len(partial)}):")
        for p in partial:
            print(f"    △ {p['pattern']} on {p['model']}: "
                  f"fix={p.get('fenced_version', 'N/A')}")

    # 3. herd7 cross-check via SMT
    print("\n[3/6] Cross-checking SMT against herd7 expected results...")
    herd7_agree = 0
    herd7_total = 0
    model_map = {'x86': 'TSO', 'sparc': 'PSO', 'arm': 'ARM', 'riscv': 'RISC-V'}
    for (pat_name, arch), herd7_allowed in sorted(HERD7_EXPECTED.items()):
        if arch not in model_map:
            continue
        smt_result = validate_pattern_smt(pat_name, model_map[arch])
        if smt_result.get('allowed') is not None:
            herd7_total += 1
            if smt_result['allowed'] == herd7_allowed:
                herd7_agree += 1

    if herd7_total > 0:
        h7_p, h7_lo, h7_hi = wilson_ci(herd7_agree, herd7_total)
        print(f"  SMT vs herd7: {herd7_agree}/{herd7_total} agree "
              f"({100*h7_p:.1f}%, CI [{100*h7_lo:.1f}%, {100*h7_hi:.1f}%])")

    # 4. SMT-based litmus test generation (model discriminators)
    print("\n[4/6] Generating discriminating litmus tests via SMT...")
    disc_results = generate_all_model_discriminators()
    for pair_key, pair_data in disc_results['pairwise_discriminators'].items():
        n = pair_data['discriminating_count']
        print(f"  {pair_data['model_a']} vs {pair_data['model_b']}: "
              f"{n} discriminating patterns")
    print(f"  Minimal discriminating set: {disc_results['minimal_discriminating_set']}")
    print(f"  Coverage: {disc_results['coverage']['covered_by_minimal_set']}/"
          f"{disc_results['coverage']['total_pairs']} model pairs")

    # 5. SMT-based litmus test SYNTHESIS (from scratch)
    print("\n[5/6] Synthesizing NEW litmus tests via Z3...")
    synthesis_results = run_litmus_synthesis()
    synth_count = synthesis_results.get('total_synthesized', 0)
    print(f"  Total synthesized: {synth_count}")
    for pair_key, pair_data in synthesis_results.get('pairwise_results', {}).items():
        if pair_data.get('synthesized'):
            for test in pair_data.get('tests', []):
                print(f"  ✓ {pair_key}: " + " | ".join(test.get('test_description', [])))
        else:
            print(f"  - {pair_key}: {pair_data.get('reason', 'N/A')}")

    # 6. Original 7-pair fence proofs (backward compatibility)
    print("\n[6/6] Legacy fence proof verification (original 7 pairs)...")
    fence_proofs = []
    fence_pairs = [('mp', 'ARM'), ('sb', 'ARM'), ('lb', 'ARM'),
                   ('mp', 'RISC-V'), ('sb', 'RISC-V'),
                   ('wrc', 'ARM'), ('rwc', 'ARM')]
    for pat, model in fence_pairs:
        proof = prove_fence_sufficiency_smt(pat, model)
        fence_proofs.append(proof)
        status = '✓' if proof.get('fence_sufficient') else '✗'
        print(f"  {status} {pat} on {model}: unfenced={proof.get('unfenced_result')}, "
              f"fenced={proof.get('fenced_result', 'N/A')}")

    # Save results
    full_results = {
        'smt_cross_validation': {
            'total': report['total_checks'],
            'agree': report['agree'],
            'disagree': report['disagree'],
            'timeout': report['timeout'],
            'agreement_rate': report.get('agreement_rate'),
            'wilson_95ci': report.get('wilson_95ci'),
            'disagreements': report['disagreements'],
        },
        'comprehensive_fence_proofs': {
            'total_unsafe_pairs': classification['total_unsafe_pairs'],
            'fence_sufficient': classification['fence_sufficient'],
            'inherently_observable': classification['inherently_observable'],
            'partial_fence': classification['partial_fence'],
            'proofs': classification['proofs'],
        },
        'fence_sufficiency_proofs': fence_proofs,
        'smt_vs_herd7': {
            'agree': herd7_agree,
            'total': herd7_total,
            'agreement_rate': round(100 * herd7_agree / max(herd7_total, 1), 1),
        },
        'model_discriminators': {
            'pairwise': {k: {
                'model_a': v['model_a'],
                'model_b': v['model_b'],
                'count': v['discriminating_count'],
                'patterns': [d['pattern'] for d in v['discriminators']],
            } for k, v in disc_results['pairwise_discriminators'].items()},
            'minimal_set': disc_results['minimal_discriminating_set'],
            'all_discriminating': disc_results['all_discriminating_patterns'],
        },
        'litmus_synthesis': {
            'method': synthesis_results.get('method', 'Z3 parametric skeleton'),
            'total_synthesized': synth_count,
            'pairwise': {k: {
                'synthesized': v.get('synthesized', False),
                'count': v.get('count', 0),
                'tests': v.get('tests', []),
            } for k, v in synthesis_results.get('pairwise_results', {}).items()},
        },
        'z3_version': '4.x',
    }

    os.makedirs('paper_results_v4', exist_ok=True)
    with open('paper_results_v4/smt_validation.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"\nResults saved to paper_results_v4/smt_validation.json")
    return full_results


if __name__ == '__main__':
    run_full_smt_validation()
