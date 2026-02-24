#!/usr/bin/env python3
"""
Head-to-head comparison of litmus-infinity with brute-force enumeration
(herd7-style) for memory model verification.

This implements the same verification algorithm as herd7 (enumerate all
candidate execution graphs, check each against model axioms) without
symmetry exploitation, and compares wall-clock time with our tool.
"""

import subprocess
import time
import json
import csv
import os
import itertools
from collections import defaultdict

# ── Litmus Test Definitions ──────────────────────────────────────────

class MemoryOp:
    def __init__(self, thread, optype, addr, value=None, reg=None):
        self.thread = thread
        self.optype = optype  # 'store' or 'load'
        self.addr = addr
        self.value = value
        self.reg = reg

class LitmusTest:
    def __init__(self, name, threads, addresses, ops, forbidden_outcome):
        self.name = name
        self.n_threads = threads
        self.n_addresses = addresses
        self.ops = ops  # list of MemoryOp
        self.forbidden = forbidden_outcome  # dict: reg -> value
        self.loads = [op for op in ops if op.optype == 'load']
        self.stores = [op for op in ops if op.optype == 'store']

# Standard litmus tests
def build_sb():
    ops = [
        MemoryOp(0, 'store', 'x', value=1),
        MemoryOp(0, 'load', 'y', reg='r0'),
        MemoryOp(1, 'store', 'y', value=1),
        MemoryOp(1, 'load', 'x', reg='r1'),
    ]
    return LitmusTest('SB', 2, 2, ops, {'r0': 0, 'r1': 0})

def build_mp():
    ops = [
        MemoryOp(0, 'store', 'x', value=1),
        MemoryOp(0, 'store', 'y', value=1),
        MemoryOp(1, 'load', 'y', reg='r0'),
        MemoryOp(1, 'load', 'x', reg='r1'),
    ]
    return LitmusTest('MP', 2, 2, ops, {'r0': 1, 'r1': 0})

def build_lb():
    ops = [
        MemoryOp(0, 'load', 'x', reg='r0'),
        MemoryOp(0, 'store', 'y', value=1),
        MemoryOp(1, 'load', 'y', reg='r1'),
        MemoryOp(1, 'store', 'x', value=1),
    ]
    return LitmusTest('LB', 2, 2, ops, {'r0': 1, 'r1': 1})

def build_iriw():
    ops = [
        MemoryOp(0, 'store', 'x', value=1),
        MemoryOp(1, 'store', 'y', value=1),
        MemoryOp(2, 'load', 'x', reg='r0'),
        MemoryOp(2, 'load', 'y', reg='r1'),
        MemoryOp(3, 'load', 'y', reg='r2'),
        MemoryOp(3, 'load', 'x', reg='r3'),
    ]
    return LitmusTest('IRIW', 4, 2, ops, {'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0})

def build_dekker():
    return build_sb()  # Structurally identical

def build_sb_n(n):
    """Build n-thread cyclic store buffering."""
    ops = []
    addrs = [f'x{i}' for i in range(n)]
    for i in range(n):
        ops.append(MemoryOp(i, 'store', addrs[i], value=1))
        ops.append(MemoryOp(i, 'load', addrs[(i+1) % n], reg=f'r{i}'))
    forbidden = {f'r{i}': 0 for i in range(n)}
    return LitmusTest(f'SB{n}', n, n, ops, forbidden)


# ── Memory Model Checking (herd7-style) ─────────────────────────────

def get_thread_ops(test, thread_id):
    return [op for op in test.ops if op.thread == thread_id]

def get_stores_to_addr(test, addr):
    """Get all stores to an address, including initial store of 0."""
    stores = [('init', addr, 0)]  # initial store
    for op in test.ops:
        if op.optype == 'store' and op.addr == addr:
            stores.append((op.thread, op.addr, op.value))
    return stores

def enumerate_rf_assignments(test):
    """Enumerate all possible reads-from assignments."""
    loads = test.loads
    if not loads:
        yield {}
        return

    # For each load, find possible stores it could read from
    choices = []
    for load in loads:
        stores = get_stores_to_addr(test, load.addr)
        choices.append([(load, store) for store in stores])

    for combo in itertools.product(*choices):
        rf = {}
        for load, store in combo:
            rf[id(load)] = store
        yield rf

def enumerate_co_assignments(test):
    """Enumerate all coherence order assignments."""
    addrs = set(op.addr for op in test.ops)
    addr_stores = {}
    for addr in addrs:
        stores = get_stores_to_addr(test, addr)
        if len(stores) > 1:
            addr_stores[addr] = stores

    if not addr_stores:
        yield {}
        return

    # For each address, enumerate total orders on stores
    addr_list = sorted(addr_stores.keys())
    addr_perms = []
    for addr in addr_list:
        stores = addr_stores[addr]
        # Initial store must be first
        non_init = [s for s in stores if s[0] != 'init']
        init = [s for s in stores if s[0] == 'init']
        perms = [init + list(p) for p in itertools.permutations(non_init)]
        addr_perms.append(perms)

    for combo in itertools.product(*addr_perms):
        co = {}
        for i, addr in enumerate(addr_list):
            co[addr] = combo[i]
        yield co

def compute_fr(rf, co):
    """Compute from-reads relation."""
    fr = []
    for load_id, store in rf.items():
        addr = store[1]
        if addr in co:
            order = co[addr]
            store_idx = None
            for i, s in enumerate(order):
                if s == store:
                    store_idx = i
                    break
            if store_idx is not None:
                for j in range(store_idx + 1, len(order)):
                    fr.append((load_id, order[j]))
    return fr

def build_po_edges(test):
    """Build program-order edges."""
    edges = []
    for t in range(test.n_threads):
        thread_ops = get_thread_ops(test, t)
        for i in range(len(thread_ops) - 1):
            edges.append((id(thread_ops[i]), id(thread_ops[i+1]),
                         thread_ops[i], thread_ops[i+1]))
    return edges

def has_cycle(edges):
    """Check if directed graph has a cycle using DFS."""
    graph = defaultdict(list)
    nodes = set()
    for src, dst in edges:
        graph[src].append(dst)
        nodes.add(src)
        nodes.add(dst)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}

    def dfs(u):
        color[u] = GRAY
        for v in graph[u]:
            if color.get(v, WHITE) == GRAY:
                return True
            if color.get(v, WHITE) == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for node in nodes:
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False

def check_sc(test, rf, co):
    """Check SC consistency: acyclic(po ∪ rf ∪ co ∪ fr)."""
    edges = []
    # po edges
    for t in range(test.n_threads):
        ops = get_thread_ops(test, t)
        for i in range(len(ops) - 1):
            edges.append((id(ops[i]), id(ops[i+1])))

    # rf edges (store -> load)
    for load_id, store in rf.items():
        if store[0] == 'init':
            edges.append(('init_' + store[1], load_id))
        else:
            store_op = [op for op in test.ops if op.optype == 'store'
                       and op.thread == store[0] and op.addr == store[1]
                       and op.value == store[2]]
            if store_op:
                edges.append((id(store_op[0]), load_id))

    # co edges
    for addr, order in co.items():
        for i in range(len(order) - 1):
            s1, s2 = order[i], order[i+1]
            n1 = 'init_' + addr if s1[0] == 'init' else id([op for op in test.ops
                  if op.optype == 'store' and op.thread == s1[0]
                  and op.addr == s1[1] and op.value == s1[2]][0])
            n2 = 'init_' + addr if s2[0] == 'init' else id([op for op in test.ops
                  if op.optype == 'store' and op.thread == s2[0]
                  and op.addr == s2[1] and op.value == s2[2]][0])
            edges.append((n1, n2))

    # fr edges
    fr = compute_fr(rf, co)
    for load_id, store in fr:
        if store[0] == 'init':
            edges.append((load_id, 'init_' + store[1]))
        else:
            store_op = [op for op in test.ops if op.optype == 'store'
                       and op.thread == store[0] and op.addr == store[1]
                       and op.value == store[2]]
            if store_op:
                edges.append((load_id, id(store_op[0])))

    return not has_cycle(edges)

def check_tso(test, rf, co):
    """Check TSO: acyclic(ppo_TSO ∪ com) where ppo excludes W->R to diff addr."""
    edges = []
    for t in range(test.n_threads):
        ops = get_thread_ops(test, t)
        for i in range(len(ops) - 1):
            for j in range(i+1, len(ops)):
                op_i, op_j = ops[i], ops[j]
                # Skip W->R to different address
                if (op_i.optype == 'store' and op_j.optype == 'load'
                    and op_i.addr != op_j.addr):
                    continue
                edges.append((id(op_i), id(op_j)))

    # Add com edges (same as SC)
    for load_id, store in rf.items():
        if store[0] == 'init':
            edges.append(('init_' + store[1], load_id))
        else:
            store_op = [op for op in test.ops if op.optype == 'store'
                       and op.thread == store[0] and op.addr == store[1]
                       and op.value == store[2]]
            if store_op:
                edges.append((id(store_op[0]), load_id))
    for addr, order in co.items():
        for i in range(len(order) - 1):
            s1, s2 = order[i], order[i+1]
            n1 = 'init_' + addr if s1[0] == 'init' else id([op for op in test.ops
                  if op.optype == 'store' and op.thread == s1[0]
                  and op.addr == s1[1] and op.value == s1[2]][0])
            n2 = 'init_' + addr if s2[0] == 'init' else id([op for op in test.ops
                  if op.optype == 'store' and op.thread == s2[0]
                  and op.addr == s2[1] and op.value == s2[2]][0])
            edges.append((n1, n2))
    fr = compute_fr(rf, co)
    for load_id, store in fr:
        if store[0] == 'init':
            edges.append((load_id, 'init_' + store[1]))
        else:
            store_op = [op for op in test.ops if op.optype == 'store'
                       and op.thread == store[0] and op.addr == store[1]
                       and op.value == store[2]]
            if store_op:
                edges.append((load_id, id(store_op[0])))

    return not has_cycle(edges)

def check_pso(test, rf, co):
    """Check PSO: additionally relaxes W->W to different addresses."""
    edges = []
    for t in range(test.n_threads):
        ops = get_thread_ops(test, t)
        for i in range(len(ops) - 1):
            for j in range(i+1, len(ops)):
                op_i, op_j = ops[i], ops[j]
                if (op_i.optype == 'store' and op_j.optype == 'load'
                    and op_i.addr != op_j.addr):
                    continue
                if (op_i.optype == 'store' and op_j.optype == 'store'
                    and op_i.addr != op_j.addr):
                    continue
                edges.append((id(op_i), id(op_j)))

    for load_id, store in rf.items():
        if store[0] == 'init':
            edges.append(('init_' + store[1], load_id))
        else:
            store_op = [op for op in test.ops if op.optype == 'store'
                       and op.thread == store[0] and op.addr == store[1]
                       and op.value == store[2]]
            if store_op:
                edges.append((id(store_op[0]), load_id))
    for addr, order in co.items():
        for i in range(len(order) - 1):
            s1, s2 = order[i], order[i+1]
            n1 = 'init_' + addr if s1[0] == 'init' else id([op for op in test.ops
                  if op.optype == 'store' and op.thread == s1[0]
                  and op.addr == s1[1] and op.value == s1[2]][0])
            n2 = 'init_' + addr if s2[0] == 'init' else id([op for op in test.ops
                  if op.optype == 'store' and op.thread == s2[0]
                  and op.addr == s2[1] and op.value == s2[2]][0])
            edges.append((n1, n2))
    fr = compute_fr(rf, co)
    for load_id, store in fr:
        if store[0] == 'init':
            edges.append((load_id, 'init_' + store[1]))
        else:
            store_op = [op for op in test.ops if op.optype == 'store'
                       and op.thread == store[0] and op.addr == store[1]
                       and op.value == store[2]]
            if store_op:
                edges.append((load_id, id(store_op[0])))

    return not has_cycle(edges)

def check_arm(test, rf, co):
    """Check ARM: only same-address load ordering preserved (simplified)."""
    edges = []
    for t in range(test.n_threads):
        ops = get_thread_ops(test, t)
        for i in range(len(ops) - 1):
            for j in range(i+1, len(ops)):
                op_i, op_j = ops[i], ops[j]
                # Only preserve same-address load-load ordering
                if op_i.addr == op_j.addr and op_i.optype == 'load' and op_j.optype == 'load':
                    edges.append((id(op_i), id(op_j)))

    # Use rfe, coe, fre instead of full com
    for load_id, store in rf.items():
        load_op = [op for op in test.ops if id(op) == load_id][0] if any(id(op) == load_id for op in test.ops) else None
        if store[0] == 'init':
            edges.append(('init_' + store[1], load_id))
        elif load_op and store[0] != load_op.thread:  # external rf only
            store_op = [op for op in test.ops if op.optype == 'store'
                       and op.thread == store[0] and op.addr == store[1]
                       and op.value == store[2]]
            if store_op:
                edges.append((id(store_op[0]), load_id))

    for addr, order in co.items():
        for i in range(len(order) - 1):
            s1, s2 = order[i], order[i+1]
            t1 = s1[0]
            t2 = s2[0]
            if t1 != t2:  # external co only
                n1 = 'init_' + addr if t1 == 'init' else id([op for op in test.ops
                      if op.optype == 'store' and op.thread == t1
                      and op.addr == s1[1] and op.value == s1[2]][0])
                n2 = 'init_' + addr if t2 == 'init' else id([op for op in test.ops
                      if op.optype == 'store' and op.thread == t2
                      and op.addr == s2[1] and op.value == s2[2]][0])
                edges.append((n1, n2))

    fr = compute_fr(rf, co)
    for load_id, store in fr:
        load_op = [op for op in test.ops if id(op) == load_id]
        if load_op and store[0] != load_op[0].thread:  # external fr
            if store[0] == 'init':
                edges.append((load_id, 'init_' + store[1]))
            else:
                store_op = [op for op in test.ops if op.optype == 'store'
                           and op.thread == store[0] and op.addr == store[1]
                           and op.value == store[2]]
                if store_op:
                    edges.append((load_id, id(store_op[0])))

    return not has_cycle(edges)


MODEL_CHECKERS = {
    'SC': check_sc,
    'TSO': check_tso,
    'PSO': check_pso,
    'ARM': check_arm,
    'RISC-V': check_arm,  # Same simplified model
}

def get_outcome(test, rf):
    """Extract the outcome (register values) from an rf assignment."""
    outcome = {}
    for load in test.loads:
        store = rf[id(load)]
        outcome[load.reg] = store[2]  # value read
    return outcome

def brute_force_verify(test, model_name):
    """Brute-force verification: enumerate all (rf, co), check consistency."""
    checker = MODEL_CHECKERS[model_name]
    n_checked = 0
    consistent_outcomes = set()
    forbidden_found = False

    for rf in enumerate_rf_assignments(test):
        outcome = get_outcome(test, rf)
        outcome_key = tuple(sorted(outcome.items()))

        for co in enumerate_co_assignments(test):
            n_checked += 1
            if checker(test, rf, co):
                consistent_outcomes.add(outcome_key)
                if outcome == test.forbidden:
                    forbidden_found = True

    is_forbidden = not forbidden_found
    return {
        'test': test.name,
        'model': model_name,
        'n_checked': n_checked,
        'n_consistent_outcomes': len(consistent_outcomes),
        'forbidden_is_forbidden': is_forbidden,
        'result': 'Forbidden' if is_forbidden else 'Allowed',
    }


# ── Comparison Runner ────────────────────────────────────────────────

def run_comparison():
    tests = {
        'SB': build_sb,
        'MP': build_mp,
        'LB': build_lb,
        'IRIW': build_iriw,
    }
    models = ['SC', 'TSO', 'PSO', 'ARM', 'RISC-V']

    results = []
    print("=" * 80)
    print("HEAD-TO-HEAD COMPARISON: Brute-Force (herd7-style) vs litmus-infinity")
    print("=" * 80)
    print()

    for test_name, builder in tests.items():
        for model in models:
            test = builder()

            # Brute-force timing
            start = time.perf_counter()
            for _ in range(100):  # 100 iterations for timing stability
                bf_result = brute_force_verify(test, model)
            bf_time = (time.perf_counter() - start) / 100 * 1000  # ms

            # litmus-infinity timing
            try:
                li_start = time.perf_counter()
                for _ in range(100):
                    proc = subprocess.run(
                        ['./target/release/litmus-cli', 'verify',
                         '--test', test_name.lower(), '--model', model],
                        capture_output=True, text=True, timeout=5,
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )
                li_time = (time.perf_counter() - li_start) / 100 * 1000
                li_result_raw = proc.stdout
                # Parse: if consistent == checked, all outcomes allowed (including forbidden one)
                import re
                consistent_m = re.search(r'Consistent:\s*(\d+)', li_result_raw)
                checked_m = re.search(r'Checked:\s*(\d+)', li_result_raw)
                if consistent_m and checked_m:
                    cons = int(consistent_m.group(1))
                    chk = int(checked_m.group(1))
                    # For the test's forbidden outcome under this model:
                    # If all outcomes are consistent, the forbidden one is also consistent → Allowed
                    # If some outcomes are forbidden, the forbidden one is forbidden → Forbidden
                    li_result = 'Allowed' if cons == chk else 'Forbidden'
                else:
                    li_result = 'Error'
            except Exception as e:
                li_time = float('nan')
                li_result = 'Error'

            result = {
                'test': test_name,
                'model': model,
                'bf_result': bf_result['result'],
                'li_result': li_result,
                'bf_time_ms': round(bf_time, 3),
                'li_time_ms': round(li_time, 3),
                'bf_graphs_checked': bf_result['n_checked'],
                'agree': bf_result['result'] == li_result,
            }
            results.append(result)

            status = "✓" if result['agree'] else "✗ MISMATCH"
            print(f"  {test_name:6s}/{model:6s}: BF={bf_result['result']:9s} ({bf_time:.3f}ms)"
                  f"  LI={li_result:9s} ({li_time:.3f}ms)  {status}")
        print()

    # Scalability comparison for n-thread SB
    print("\n" + "=" * 80)
    print("SCALABILITY: n-thread SB under SC (brute-force vs symmetry-compressed)")
    print("=" * 80)

    scalability_results = []
    for n in range(2, 7):
        test = build_sb_n(n)
        start = time.perf_counter()
        n_iters = max(1, 1000 // (2**n))
        for _ in range(n_iters):
            bf_result = brute_force_verify(test, 'SC')
        bf_time = (time.perf_counter() - start) / n_iters * 1000

        try:
            li_start = time.perf_counter()
            for _ in range(n_iters):
                proc = subprocess.run(
                    ['./target/release/litmus-cli', 'verify',
                     '--test', f'sb{n}' if n > 2 else 'sb', '--model', 'SC'],
                    capture_output=True, text=True, timeout=30,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
            li_time = (time.perf_counter() - li_start) / n_iters * 1000
        except:
            li_time = float('nan')

        sr = {
            'n': n,
            'outcomes': 2**n,
            'bf_time_ms': round(bf_time, 3),
            'li_time_ms': round(li_time, 3),
            'bf_graphs': bf_result['n_checked'],
        }
        scalability_results.append(sr)
        print(f"  n={n}: outcomes={2**n:5d}  BF={bf_time:8.3f}ms  "
              f"LI={li_time:8.3f}ms  BF_graphs={bf_result['n_checked']}")

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'herd7_comparison.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['test', 'model', 'bf_result',
                                               'li_result', 'bf_time_ms',
                                               'li_time_ms', 'bf_graphs_checked',
                                               'agree'])
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(output_dir, 'scalability_comparison.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n', 'outcomes', 'bf_time_ms',
                                               'li_time_ms', 'bf_graphs'])
        writer.writeheader()
        writer.writerows(scalability_results)

    # Summary
    n_agree = sum(1 for r in results if r['agree'])
    n_total = len(results)
    print(f"\n{'='*80}")
    print(f"SUMMARY: {n_agree}/{n_total} test-model pairs agree between methods")

    bf_faster = sum(1 for r in results if r['bf_time_ms'] < r['li_time_ms'])
    print(f"Brute-force faster on {bf_faster}/{n_total} pairs (expected for small tests)")
    print(f"Results saved to {output_dir}/herd7_comparison.csv")

    return results, scalability_results

if __name__ == '__main__':
    run_comparison()
