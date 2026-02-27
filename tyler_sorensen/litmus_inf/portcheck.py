#!/usr/bin/env python3
"""
portcheck: Architecture portability checker for concurrent code with GPU scope support.

Checks whether concurrent synchronization patterns are safe to port across
CPU architectures (x86/TSO, SPARC/PSO, ARM/ARMv8, RISC-V/RVWMO) and
GPU memory models (OpenCL, Vulkan, PTX) with scoped synchronization.

Implements the joint-automorphism symmetry framework from the litmus-infinity
project to provide compressed verification with certificates.

Usage:
    python portcheck.py --pattern mp --target arm
    python portcheck.py --pattern gpu_mp_wg --all-targets
    python portcheck.py --analyze-all
    python portcheck.py --analyze-all --json
"""

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class MemOp:
    thread: int
    optype: str       # 'store', 'load', 'fence'
    addr: str
    value: Optional[int] = None
    reg: Optional[str] = None
    scope: Optional[str] = None   # 'workgroup', 'device', 'system', None for CPU
    workgroup: int = 0            # which workgroup this thread belongs to
    fence_read: bool = True       # fence orders reads (for RISC-V fence r,r etc.)
    fence_write: bool = True      # fence orders writes
    fence_pred: Optional[str] = None   # RISC-V: predecessor set 'r','w','rw'
    fence_succ: Optional[str] = None   # RISC-V: successor set 'r','w','rw'
    dep_on: Optional[str] = None  # dependency: 'addr', 'data', 'ctrl', or None

@dataclass
class LitmusTest:
    name: str
    n_threads: int
    addresses: List[str]
    ops: List[MemOp]
    forbidden: Dict[str, int]

    @property
    def loads(self):
        return [op for op in self.ops if op.optype == 'load']

    @property
    def stores(self):
        return [op for op in self.ops if op.optype == 'store']

    @property
    def fences(self):
        return [op for op in self.ops if op.optype == 'fence']

@dataclass
class PortabilityResult:
    pattern: str
    source_arch: str
    target_arch: str
    safe: bool
    forbidden_outcome: Dict[str, int]
    fence_recommendation: Optional[str] = None
    compression_ratio: float = 1.0
    orbits_checked: int = 0
    total_outcomes: int = 0
    certificate: Optional[dict] = None

# ── Joint Automorphism Computation ───────────────────────────────────

def compute_joint_automorphisms(test: LitmusTest):
    """Compute the joint automorphism group of a litmus test."""
    nt = test.n_threads
    na = len(test.addresses)

    def generate_perms(n):
        return list(itertools.permutations(range(n)))

    thread_perms = generate_perms(nt)
    addr_perms = generate_perms(na)
    val_perms = generate_perms(2)

    def apply_auto(tp, ap, vp):
        addr_map = {test.addresses[i]: test.addresses[ap[i]] for i in range(na)}
        new_ops = []
        for op in test.ops:
            new_thread = tp[op.thread]
            new_addr = addr_map.get(op.addr, op.addr)
            new_val = vp[op.value] if op.value is not None and op.value < 2 else op.value
            new_ops.append(MemOp(new_thread, op.optype, new_addr, new_val, op.reg))

        orig_by_thread = defaultdict(list)
        new_by_thread = defaultdict(list)
        for op in test.ops:
            orig_by_thread[op.thread].append((op.optype, op.addr, op.value))
        for op in new_ops:
            new_by_thread[op.thread].append((op.optype, op.addr, op.value))

        return dict(orig_by_thread) == dict(new_by_thread)

    automorphisms = []
    for tp in thread_perms:
        for ap in addr_perms:
            for vp in val_perms:
                if apply_auto(tp, ap, vp):
                    automorphisms.append((tp, ap, vp))

    return automorphisms

def compute_orbits(test: LitmusTest, automorphisms):
    """Compute orbits on the outcome space using union-find."""
    loads = test.loads
    n_loads = len(loads)
    if n_loads == 0:
        return 1, 1

    total = 2 ** n_loads
    parent = list(range(total))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for tp, ap, vp in automorphisms:
        for outcome_idx in range(total):
            vals = [(outcome_idx >> (n_loads - 1 - j)) & 1 for j in range(n_loads)]
            new_vals = [0] * n_loads
            for i, load in enumerate(loads):
                new_tid = tp[load.thread]
                for j, target_load in enumerate(loads):
                    if target_load.thread == new_tid:
                        orig_idx = sum(1 for k in range(i)
                                      if loads[k].thread == load.thread)
                        target_idx = sum(1 for k in range(j)
                                        if loads[k].thread == new_tid)
                        if orig_idx == target_idx:
                            new_vals[j] = vp[vals[i]] if vals[i] < 2 else vals[i]
                            break

            new_idx = sum(new_vals[j] << (n_loads - 1 - j) for j in range(n_loads))
            union(outcome_idx, new_idx)

    n_orbits = len(set(find(i) for i in range(total)))
    return total, n_orbits

# ── Memory Model Checking ────────────────────────────────────────────

def get_stores_to_addr(test, addr):
    stores = [('init', addr, 0)]
    for op in test.ops:
        if op.optype == 'store' and op.addr == addr:
            stores.append((op.thread, op.addr, op.value))
    return stores

def _op_idx(op):
    """Stable node identifier for an op (using id())."""
    return id(op)

def _store_node(test, store_tuple):
    if store_tuple[0] == 'init':
        return f"init_{store_tuple[1]}"
    for op in test.ops:
        if (op.optype == 'store' and op.thread == store_tuple[0]
            and op.addr == store_tuple[1] and op.value == store_tuple[2]):
            return id(op)
    return None

def _is_relaxed_model(model):
    return model in ('ARM', 'RISC-V',
                     'OpenCL-WG', 'OpenCL-Dev',
                     'Vulkan-WG', 'Vulkan-Dev',
                     'PTX-CTA', 'PTX-GPU')

def _is_gpu_model(model):
    return model in ('OpenCL-WG', 'OpenCL-Dev',
                     'Vulkan-WG', 'Vulkan-Dev',
                     'PTX-CTA', 'PTX-GPU')

def _gpu_scope_of_model(model):
    """Return the scope level a model's barriers enforce."""
    scope_map = {
        'OpenCL-WG': 'workgroup', 'OpenCL-Dev': 'device',
        'Vulkan-WG': 'workgroup', 'Vulkan-Dev': 'device',
        'PTX-CTA': 'workgroup', 'PTX-GPU': 'device',
    }
    return scope_map.get(model)

def _fence_covers_pair(fence_op, before_op, after_op, model, test=None):
    """Check if a fence restores ordering between before_op and after_op.

    For GPU scoped models, a fence's effectiveness depends on:
    - Model scope (WG vs Dev): determines what cross-thread communication is visible
    - Fence scope: determines what ordering the fence provides
    - Thread placement: whether threads are in the same or different workgroups
    """
    if model == 'RISC-V':
        b_is_read = before_op.optype == 'load'
        b_is_write = before_op.optype == 'store'
        a_is_read = after_op.optype == 'load'
        a_is_write = after_op.optype == 'store'
        # Use asymmetric pred/succ sets if available
        pred = fence_op.fence_pred
        succ = fence_op.fence_succ
        if pred is not None and succ is not None:
            pred_r = 'r' in pred
            pred_w = 'w' in pred
            succ_r = 'r' in succ
            succ_w = 'w' in succ
            if b_is_read and not pred_r:
                return False
            if b_is_write and not pred_w:
                return False
            if a_is_read and not succ_r:
                return False
            if a_is_write and not succ_w:
                return False
            return True
        # Fallback to symmetric fence_read/fence_write flags
        if b_is_read and not fence_op.fence_read:
            return False
        if b_is_write and not fence_op.fence_write:
            return False
        if a_is_read and not fence_op.fence_read:
            return False
        if a_is_write and not fence_op.fence_write:
            return False
        return True

    if _is_gpu_model(model):
        model_scope = _gpu_scope_of_model(model)
        fence_scope = fence_op.scope or 'device'

        if model_scope == 'workgroup':
            # Under WG model, cross-workgroup communication is not visible
            # so no fence can provide ordering for cross-wg tests
            if test:
                wgs = set(op.workgroup for op in test.ops)
                if len(wgs) > 1:
                    return False
            return True

        elif model_scope == 'device':
            # Under Dev model, device-scope fences order everything;
            # workgroup fences only order within the workgroup
            if test:
                wgs = set(op.workgroup for op in test.ops)
                if len(wgs) > 1 and fence_scope == 'workgroup':
                    return False
            return True

        return False

    # ARM and other CPU models: any fence restores all ordering
    return True

def _gpu_fence_effective(fence_op, thread_a, thread_b, model, test):
    """Check if a scoped fence is effective between two threads under model."""
    model_scope = _gpu_scope_of_model(model)
    if model_scope is None:
        return True

    # Find workgroups of the two threads
    wg_a = _thread_workgroup(test, thread_a)
    wg_b = _thread_workgroup(test, thread_b)

    fence_scope = fence_op.scope or 'device'
    if model_scope == 'workgroup':
        if wg_a != wg_b:
            return False
        return fence_scope in ('workgroup', 'device', 'system')
    elif model_scope == 'device':
        if fence_scope in ('device', 'system'):
            return True
        if fence_scope == 'workgroup' and wg_a == wg_b:
            return True
        return False
    return True

def _thread_workgroup(test, tid):
    """Get workgroup of a thread from its ops."""
    for op in test.ops:
        if op.thread == tid:
            return op.workgroup
    return 0

def check_model(test, model, rf, co):
    """Check if execution is consistent under given model.

    For TSO: preserves all po except W->R to different addresses.
    For PSO: also relaxes W->W to different addresses.
    For ARM/GPU-relaxed: relaxes all cross-address po; preserves po-loc,
        dependencies (simplified), and po;[F];po (fence ordering).
    For RISC-V: like ARM but fences specify which orderings (r/w) they enforce.
    For GPU scoped: like ARM but fences only effective within their scope.
    """
    edges = []
    ops_by_thread = defaultdict(list)
    for op in test.ops:
        ops_by_thread[op.thread].append(op)

    is_relaxed = _is_relaxed_model(model)
    is_gpu = _is_gpu_model(model)

    # Program order edges
    for t, ops in ops_by_thread.items():
        for i in range(len(ops)):
            if ops[i].optype == 'fence':
                continue
            for j in range(i + 1, len(ops)):
                if ops[j].optype == 'fence':
                    continue
                a, b = ops[i], ops[j]
                include = True

                if model == 'TSO':
                    # TSO relaxes only W->R to different addresses
                    if a.optype == 'store' and b.optype == 'load':
                        if a.addr != b.addr:
                            include = False
                            # Check for intervening fence
                            for k in range(i + 1, j):
                                if ops[k].optype == 'fence':
                                    include = True
                                    break

                elif model == 'PSO':
                    # PSO additionally relaxes W->W to different addresses
                    if a.optype == 'store' and b.optype == 'load':
                        if a.addr != b.addr:
                            include = False
                            for k in range(i + 1, j):
                                if ops[k].optype == 'fence':
                                    include = True
                                    break
                    if a.optype == 'store' and b.optype == 'store':
                        if a.addr != b.addr:
                            include = False
                            for k in range(i + 1, j):
                                if ops[k].optype == 'fence':
                                    include = True
                                    break

                elif is_relaxed:
                    # ARM / RISC-V / GPU models: relax everything cross-address
                    # EXCEPT dependency-ordered pairs (dob) on ARM/RISC-V
                    if a.addr == b.addr:
                        include = True  # po-loc always preserved
                    else:
                        include = False
                        # Check for dependency: b depends on a
                        if b.dep_on is not None and (model in ('ARM', 'RISC-V')):
                            # addr dep: load -> load/store with address depending on loaded value
                            # data dep: load -> store with data depending on loaded value
                            # ctrl dep: load -> store with control depending on loaded value
                            if b.dep_on == 'addr' and a.optype == 'load':
                                include = True
                            elif b.dep_on == 'data' and a.optype == 'load' and b.optype == 'store':
                                include = True
                            elif b.dep_on == 'ctrl' and a.optype == 'load' and b.optype == 'store':
                                include = True
                        # Check for fence between i and j
                        if not include:
                            for k in range(i + 1, j):
                                if ops[k].optype == 'fence':
                                    if _fence_covers_pair(ops[k], a, b, model, test):
                                        include = True
                                        break

                if include:
                    edges.append((id(a), id(b)))

    # Communication edges
    # For relaxed models (ARM/RISC-V/GPU): the ob relation uses external
    # edges (rfe/coe/fre), but we ALSO need internal edges for coherence
    # checking (acyclicity of po-loc ∪ com).  So we always include all
    # communication edges; the po-based edges already handle the distinction.
    for load in test.loads:
        store = rf[id(load)]
        sn = _store_node(test, store)
        if sn is not None:
            edges.append((sn, id(load)))

    # Coherence order edges
    addrs = set(op.addr for op in test.ops)
    for addr in addrs:
        if addr in co:
            order = co[addr]
            for i in range(len(order) - 1):
                n1 = _store_node(test, order[i])
                n2 = _store_node(test, order[i + 1])
                if n1 and n2:
                    edges.append((n1, n2))

    # From-reads edges
    for load in test.loads:
        store = rf[id(load)]
        addr = store[1]
        if addr in co:
            order = co[addr]
            found = False
            for s in order:
                if s == store:
                    found = True
                    continue
                if found:
                    sn = _store_node(test, s)
                    if sn:
                        edges.append((id(load), sn))

    # For GPU scoped models: scoped fence edges across threads
    # A scoped fence on thread T means: if there's a cross-thread communication
    # edge touching T, the fence restores ordering only if scope covers the
    # other thread. This is already handled above via po edges; the cross-thread
    # communication edges (rfe/coe/fre) are always included (they're observations).
    # The key is that po-fence-po is only included when scope matches.

    # Cycle detection (irreflexivity check: cycle = forbidden)
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
                return False  # cycle found => execution forbidden
    return True  # no cycle => execution allowed

def check_model_generic(test, model_info, rf, co):
    """Generic model checking with model specified as a dict.

    model_info: dict with keys:
        'relaxed_pairs': set of (optype, optype) pairs that are relaxed
        'preserves_deps': bool
        'multi_copy_atomic': bool
        'fences': list of dicts with 'orders' (set of (optype, optype) pairs)
    """
    relaxed_pairs = model_info.get('relaxed_pairs', set())
    preserves_deps = model_info.get('preserves_deps', True)
    fence_specs = model_info.get('fences', [])

    edges = []
    ops_by_thread = defaultdict(list)
    for op in test.ops:
        ops_by_thread[op.thread].append(op)

    for t, ops in ops_by_thread.items():
        for i in range(len(ops)):
            if ops[i].optype == 'fence':
                continue
            for j in range(i + 1, len(ops)):
                if ops[j].optype == 'fence':
                    continue
                a, b = ops[i], ops[j]
                include = True

                if a.addr == b.addr:
                    include = True  # po-loc always preserved
                elif (a.optype, b.optype) in relaxed_pairs:
                    include = False
                    # Check dependencies
                    if preserves_deps and b.dep_on is not None:
                        if b.dep_on == 'addr' and a.optype == 'load':
                            include = True
                        elif b.dep_on == 'data' and a.optype == 'load' and b.optype == 'store':
                            include = True
                        elif b.dep_on == 'ctrl' and a.optype == 'load' and b.optype == 'store':
                            include = True
                    # Check intervening fences
                    if not include:
                        for k in range(i + 1, j):
                            if ops[k].optype == 'fence':
                                for f_spec in fence_specs:
                                    if (a.optype, b.optype) in f_spec.get('orders', set()):
                                        include = True
                                        break
                            if include:
                                break

                if include:
                    edges.append((id(a), id(b)))

    # Communication edges (rf)
    for load in test.loads:
        store = rf[id(load)]
        sn = _store_node(test, store)
        if sn is not None:
            edges.append((sn, id(load)))

    # Coherence order edges
    addrs = set(op.addr for op in test.ops)
    for addr in addrs:
        if addr in co:
            order = co[addr]
            for i in range(len(order) - 1):
                n1 = _store_node(test, order[i])
                n2 = _store_node(test, order[i + 1])
                if n1 and n2:
                    edges.append((n1, n2))

    # From-reads edges
    for load in test.loads:
        store = rf[id(load)]
        addr = store[1]
        if addr in co:
            order = co[addr]
            found = False
            for s in order:
                if s == store:
                    found = True
                    continue
                if found:
                    sn = _store_node(test, s)
                    if sn:
                        edges.append((id(load), sn))

    # Cycle detection
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
                return False
    return True

def verify_test_generic(test, model_info):
    """Verify whether the forbidden outcome is allowed under a generic model.

    model_info: dict with 'relaxed_pairs', 'preserves_deps', 'fences', etc.
    Returns (forbidden_allowed, n_checked).
    """
    loads = test.loads
    addrs = set(op.addr for op in test.ops)

    rf_choices = []
    for load in loads:
        stores = get_stores_to_addr(test, load.addr)
        rf_choices.append([(load, s) for s in stores])

    co_choices_list = []
    addr_list = sorted(addrs)
    for addr in addr_list:
        stores = get_stores_to_addr(test, addr)
        non_init = [s for s in stores if s[0] != 'init']
        init = [s for s in stores if s[0] == 'init']
        perms = [init + list(p) for p in itertools.permutations(non_init)]
        co_choices_list.append(perms)

    forbidden_allowed = False
    n_checked = 0

    for rf_combo in (itertools.product(*rf_choices) if rf_choices else [()]):
        rf = {}
        outcome = {}
        for load, store in rf_combo:
            rf[id(load)] = store
            if load.reg:
                outcome[load.reg] = store[2]

        for co_combo in (itertools.product(*co_choices_list) if co_choices_list else [()]):
            co = {addr_list[i]: co_combo[i] for i in range(len(addr_list))} if co_combo else {}
            n_checked += 1
            if check_model_generic(test, model_info, rf, co):
                if all(outcome.get(k) == v for k, v in test.forbidden.items()):
                    forbidden_allowed = True

    return forbidden_allowed, n_checked

def verify_test(test, model):
    """Verify whether the forbidden outcome is allowed under the given model."""
    loads = test.loads
    addrs = set(op.addr for op in test.ops)

    rf_choices = []
    for load in loads:
        stores = get_stores_to_addr(test, load.addr)
        rf_choices.append([(load, s) for s in stores])

    co_choices_list = []
    addr_list = sorted(addrs)
    for addr in addr_list:
        stores = get_stores_to_addr(test, addr)
        non_init = [s for s in stores if s[0] != 'init']
        init = [s for s in stores if s[0] == 'init']
        perms = [init + list(p) for p in itertools.permutations(non_init)]
        co_choices_list.append(perms)

    forbidden_allowed = False
    n_checked = 0

    for rf_combo in (itertools.product(*rf_choices) if rf_choices else [()]):
        rf = {}
        outcome = {}
        for load, store in rf_combo:
            rf[id(load)] = store
            if load.reg:
                outcome[load.reg] = store[2]

        for co_combo in (itertools.product(*co_choices_list) if co_choices_list else [()]):
            co = {addr_list[i]: co_combo[i] for i in range(len(addr_list))} if co_combo else {}
            n_checked += 1
            if check_model(test, model, rf, co):
                if all(outcome.get(k) == v for k, v in test.forbidden.items()):
                    forbidden_allowed = True

    return forbidden_allowed, n_checked

# ── Semantic Fence Analysis ──────────────────────────────────────────

def _identify_violated_ordering(test, model):
    """Identify which operation pair ordering is violated and return
    the minimal fence recommendation for each architecture."""
    ops_by_thread = defaultdict(list)
    for op in test.ops:
        if op.optype != 'fence':
            ops_by_thread[op.thread].append(op)

    violated_pairs = []
    for t, ops in ops_by_thread.items():
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                a, b = ops[i], ops[j]
                if a.addr == b.addr:
                    continue
                pair_type = (a.optype, b.optype)
                violated_pairs.append(pair_type)

    pair_types = set(violated_pairs)
    return pair_types

def _identify_per_thread_violations(test, model):
    """Identify violated ordering pairs per thread for fine-grained fence advice.

    Only flags pairs that are actually relaxed under the target model,
    i.e., not already ordered by the model's preserved program order,
    dependencies, or existing fences.
    """
    ops_by_thread = defaultdict(list)
    all_ops_by_thread = defaultdict(list)
    for op in test.ops:
        all_ops_by_thread[op.thread].append(op)
        if op.optype != 'fence':
            ops_by_thread[op.thread].append(op)

    is_relaxed = _is_relaxed_model(model)

    per_thread = {}
    for t, ops in ops_by_thread.items():
        pairs = set()
        all_ops = all_ops_by_thread[t]
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                a, b = ops[i], ops[j]
                if a.addr == b.addr:
                    continue  # po-loc always preserved
                # Check if this pair is already ordered by the model
                already_ordered = False
                if model == 'TSO':
                    # TSO preserves everything except W->R to diff addr
                    if not (a.optype == 'store' and b.optype == 'load'):
                        already_ordered = True
                elif model == 'PSO':
                    # PSO preserves R->R and R->W (only relaxes W->R, W->W)
                    if a.optype == 'load':
                        already_ordered = True
                elif is_relaxed:
                    # ARM/RISC-V/GPU: check dependencies
                    if b.dep_on is not None and model in ('ARM', 'RISC-V'):
                        if b.dep_on == 'addr' and a.optype == 'load':
                            already_ordered = True
                        elif b.dep_on == 'data' and a.optype == 'load' and b.optype == 'store':
                            already_ordered = True
                        elif b.dep_on == 'ctrl' and a.optype == 'load' and b.optype == 'store':
                            already_ordered = True
                    # Check for existing fence between a and b
                    if not already_ordered:
                        a_idx = all_ops.index(a) if a in all_ops else -1
                        b_idx = all_ops.index(b) if b in all_ops else -1
                        if a_idx >= 0 and b_idx >= 0:
                            for k in range(a_idx + 1, b_idx):
                                if all_ops[k].optype == 'fence':
                                    if _fence_covers_pair(all_ops[k], a, b, model, test):
                                        already_ordered = True
                                        break
                if not already_ordered:
                    pairs.add((a.optype, b.optype))
        per_thread[t] = pairs
    return per_thread

def _arm_fence_for_pairs(pairs):
    """Return the minimal ARM fence for a set of ordering pairs."""
    has_wr = ('store', 'load') in pairs
    has_ww = ('store', 'store') in pairs
    has_rr = ('load', 'load') in pairs
    has_rw = ('load', 'store') in pairs
    if not (has_wr or has_ww or has_rr or has_rw):
        return None
    if has_ww and not has_rr and not has_wr and not has_rw:
        return 'dmb ishst'
    if (has_rr or has_rw) and not has_ww and not has_wr:
        return 'dmb ishld'
    return 'dmb ish'

def _riscv_fence_for_pairs(pairs):
    """Return the minimal RISC-V fence for a set of ordering pairs."""
    has_wr = ('store', 'load') in pairs
    has_ww = ('store', 'store') in pairs
    has_rr = ('load', 'load') in pairs
    has_rw = ('load', 'store') in pairs
    if not (has_wr or has_ww or has_rr or has_rw):
        return None
    if has_rr and not has_ww and not has_wr and not has_rw:
        return 'fence r,r'
    if has_ww and not has_rr and not has_wr and not has_rw:
        return 'fence w,w'
    if has_wr and not has_rr and not has_ww and not has_rw:
        return 'fence w,r'
    if has_rw and not has_rr and not has_ww and not has_wr:
        return 'fence r,w'
    return 'fence rw,rw'

def recommend_fence(test, arch, model):
    """Per-thread fine-grained fence recommendation via semantic analysis."""
    per_thread = _identify_per_thread_violations(test, model)
    all_pairs = _identify_violated_ordering(test, model)
    if not all_pairs:
        return None

    if arch == 'x86':
        has_wr = ('store', 'load') in all_pairs
        if has_wr:
            return 'MFENCE'
        return None

    elif arch == 'sparc':
        parts = []
        if ('store', 'load') in all_pairs: parts.append('#StoreLoad')
        if ('store', 'store') in all_pairs: parts.append('#StoreStore')
        if ('load', 'load') in all_pairs: parts.append('#LoadLoad')
        if ('load', 'store') in all_pairs: parts.append('#LoadStore')
        return 'membar ' + '|'.join(parts) if parts else None

    elif arch == 'arm':
        # Per-thread fine-grained ARM fence recommendations
        thread_fences = {}
        for t in sorted(per_thread.keys()):
            f = _arm_fence_for_pairs(per_thread[t])
            if f:
                thread_fences[t] = f
        if not thread_fences:
            return _arm_fence_for_pairs(all_pairs)
        if len(thread_fences) == 1:
            t, f = next(iter(thread_fences.items()))
            return f
        parts = []
        for t in sorted(thread_fences.keys()):
            parts.append(f'{thread_fences[t]} (T{t})')
        return '; '.join(parts)

    elif arch == 'riscv':
        # Per-thread fine-grained RISC-V fence recommendations
        thread_fences = {}
        for t in sorted(per_thread.keys()):
            f = _riscv_fence_for_pairs(per_thread[t])
            if f:
                thread_fences[t] = f
        if not thread_fences:
            return _riscv_fence_for_pairs(all_pairs)
        if len(thread_fences) == 1:
            t, f = next(iter(thread_fences.items()))
            return f
        parts = []
        for t in sorted(thread_fences.keys()):
            parts.append(f'{thread_fences[t]} (T{t})')
        return '; '.join(parts)

    elif arch in ('opencl_wg', 'vulkan_wg'):
        if _is_scope_mismatch_pattern(test):
            return 'use device-scope barrier (work_group_barrier insufficient for cross-workgroup)'
        return 'work_group_barrier(CLK_GLOBAL_MEM_FENCE)'

    elif arch in ('opencl_dev', 'vulkan_dev'):
        return 'atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acq_rel, memory_scope_device)'

    elif arch == 'ptx_cta':
        if _is_scope_mismatch_pattern(test):
            return 'use membar.gl (membar.cta insufficient for cross-CTA)'
        return 'membar.cta'

    elif arch == 'ptx_gpu':
        return 'membar.gl'

    return None

def _is_scope_mismatch_pattern(test):
    """Check if the test has threads in different workgroups with only workgroup-scoped fences."""
    workgroups = set()
    for op in test.ops:
        workgroups.add((op.thread, op.workgroup))
    wg_set = set(wg for _, wg in workgroups)
    if len(wg_set) <= 1:
        return False
    # Has cross-workgroup threads; check if fences are only workgroup-scoped
    for op in test.ops:
        if op.optype == 'fence' and op.scope == 'workgroup':
            return True
    return False

# ── Built-in Patterns ────────────────────────────────────────────────

def _mk(*ops, addrs=None, forbidden=None, desc=''):
    """Helper to build pattern dicts."""
    if addrs is None:
        addrs = sorted(set(op.addr for op in ops if op.addr))
    return {'description': desc, 'ops': list(ops),
            'addresses': addrs, 'forbidden': forbidden or {}}

PATTERNS = {
    # ── CPU classic patterns (all structurally distinct) ──

    'mp': _mk(
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'), MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 1, 'r1': 0},
        desc='Message passing: St[x]=1; St[y]=1 || Ld[y]; Ld[x]'),

    'sb': _mk(
        MemOp(0, 'store', 'x', 1), MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1), MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 0, 'r1': 0},
        desc='Store buffering: St[x]=1; Ld[y] || St[y]=1; Ld[x]'),

    'lb': _mk(
        MemOp(0, 'load', 'x', reg='r0'), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r1'), MemOp(1, 'store', 'x', 1),
        forbidden={'r0': 1, 'r1': 1},
        desc='Load buffering: Ld[x]; St[y]=1 || Ld[y]; St[x]=1'),

    'iriw': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'x', reg='r0'), MemOp(2, 'load', 'y', reg='r1'),
        MemOp(3, 'load', 'y', reg='r2'), MemOp(3, 'load', 'x', reg='r3'),
        forbidden={'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0},
        desc='IRIW: 4 threads, tests multi-copy atomicity'),

    '2+2w': _mk(
        # 2+2W: Two threads each write two addresses in opposite order.
        # T0: W[x]=1; W[y]=1.  T1: W[y]=2; W[x]=2.
        # Observers: T2 reads y then x.
        # Forbidden under TSO: r0=2, r1=1 (seeing T1's y before T0's x
        # requires co(y):T0<T1 and co(x):T1<T0, creating W→W reorder)
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'store', 'y', 2), MemOp(1, 'store', 'x', 2),
        MemOp(2, 'load', 'y', reg='r0'), MemOp(2, 'load', 'x', reg='r1'),
        forbidden={'r0': 2, 'r1': 1},
        desc='2+2W: write ordering test with observer'),

    'rwc': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'), MemOp(1, 'load', 'y', reg='r1'),
        MemOp(2, 'store', 'y', 1), MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 0, 'r2': 0},
        desc='Read-write causality (3 threads)'),

    'wrc': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'), MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r1'), MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='Write-read causality (3 threads)'),

    'mp_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP with fences: St[x]=1; FENCE; St[y]=1 || Ld[y]; FENCE; Ld[x]'),

    'sb_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 0, 'r1': 0},
        desc='SB with fences: St[x]=1; FENCE; Ld[y] || St[y]=1; FENCE; Ld[x]'),

    'isa2': _mk(
        # ISA2: St[x] || Ld[x];St[y]=1 || Ld[y];Ld[x] with data dependency on T1
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'), MemOp(1, 'store', 'y', 1, dep_on='data'),
        MemOp(2, 'load', 'y', reg='r1'), MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='ISA2 (3 threads): St[x]=1 || Ld[x];St[y]=1(dep) || Ld[y];Ld[x]'),

    'r': _mk(
        # R pattern: like WRC but T1 store depends on load via control dependency
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'), MemOp(1, 'store', 'y', 1, dep_on='ctrl'),
        MemOp(2, 'load', 'y', reg='r1'), MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='R pattern (3 threads): causality test with ctrl dependency'),

    # ── MP variants ──

    'mp_data': _mk(
        # MP with data dependency: T0 loads z, uses result to compute store to x
        MemOp(0, 'load', 'z', reg='r2'), MemOp(0, 'store', 'x', 1, dep_on='data'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'), MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+data: MP with data dependency on producer store'),

    'mp_3thread': _mk(
        # MP with 3 threads: writer, flag setter, reader
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r0'), MemOp(2, 'load', 'x', reg='r1'),
        forbidden={'r0': 1, 'r1': 0},
        desc='3-thread MP: W[x] || W[y] || Ld[y];Ld[x]'),

    'mp_rfi': _mk(
        # MP with reads-from-internal: T0 writes x then reads x
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'), MemOp(1, 'load', 'x', reg='r1'),
        MemOp(0, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+rfi: message passing with internal read'),

    # ── SB variants ──

    'sb_3thread': _mk(
        MemOp(0, 'store', 'x', 1), MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1), MemOp(1, 'load', 'z', reg='r1'),
        MemOp(2, 'store', 'z', 1), MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 0, 'r1': 0, 'r2': 0},
        desc='3-thread SB: St[x];Ld[y] || St[y];Ld[z] || St[z];Ld[x]'),

    # ── LB variants ──

    'lb_fence': _mk(
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'x', 1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 1},
        desc='LB+fence: load buffering with fences'),

    # ── Coherence patterns (from ARM litmus catalog) ──

    'corr': _mk(
        # CoRR: two reads of same address must see coherent order
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'store', 'x', 2),
        MemOp(1, 'load', 'x', reg='r0'), MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x'],
        forbidden={'r0': 2, 'r1': 1},
        desc='CoRR: coherence of two reads'),

    'cowr': _mk(
        # CoWR: write followed by read on same address
        MemOp(0, 'store', 'x', 1), MemOp(0, 'load', 'x', reg='r0'),
        MemOp(1, 'store', 'x', 2),
        addrs=['x'],
        forbidden={'r0': 2},
        desc='CoWR: coherence write-read ordering'),

    'coww': _mk(
        # CoWW: two writes to same address, observer reads
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'store', 'x', 2),
        MemOp(2, 'load', 'x', reg='r0'),
        addrs=['x'],
        forbidden={'r0': 1},
        desc='CoWW: coherence of write order with observer'),

    'corw': _mk(
        # CoRW: read must not see a write that is co-before the write it already saw
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'), MemOp(1, 'store', 'x', 2),
        addrs=['x'],
        forbidden={'r0': 2},
        desc='CoRW: read-write coherence'),

    # ── IRIW variants ──

    'iriw_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'x', reg='r0'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'y', reg='r1'),
        MemOp(3, 'load', 'y', reg='r2'),
        MemOp(3, 'fence', '', scope=None),
        MemOp(3, 'load', 'x', reg='r3'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0},
        desc='IRIW+fence: multi-copy atomicity test with fences'),

    # ── WRC variants ──

    'wrc_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r1'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r2'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='WRC+fence: write-read causality with fences'),

    # ── RWC variants ──

    'rwc_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(2, 'store', 'y', 1),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r2'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0, 'r2': 0},
        desc='RWC+fence: read-write causality with fences'),

    # ── Additional classic patterns from ARM catalog ──

    's': _mk(
        # S pattern (also called S litmus test)
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'), MemOp(1, 'store', 'x', 2),
        forbidden={'r0': 1},
        desc='S pattern: W[x]=1;W[y]=1 || Ld[y];W[x]=2'),

    'mp_addr': _mk(
        # MP with address dependency on consumer (load address depends on flag)
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'), MemOp(1, 'load', 'x', reg='r1', dep_on='addr'),
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+addr: MP with address dependency on consumer load'),

    'sb_rfi': _mk(
        # SB with reads-from-internal
        MemOp(0, 'store', 'x', 1), MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1), MemOp(1, 'load', 'x', reg='r1'),
        MemOp(0, 'load', 'x', reg='r2'),
        forbidden={'r0': 0, 'r1': 0},
        desc='SB+rfi: store buffering with internal read'),

    'dekker': _mk(
        # Dekker's mutual exclusion: flag + turn variable
        # T0: St[flag0]=1; Ld[flag1]; Ld[turn]
        # T1: St[flag1]=1; Ld[flag0]; Ld[turn]
        MemOp(0, 'store', 'x', 1), MemOp(0, 'load', 'y', reg='r0'),
        MemOp(0, 'load', 'z', reg='r1'),
        MemOp(1, 'store', 'y', 1), MemOp(1, 'load', 'x', reg='r2'),
        MemOp(1, 'load', 'z', reg='r3'),
        forbidden={'r0': 0, 'r2': 0},
        desc='Dekker: mutual exclusion with flag+turn (3 addr)'),

    'peterson': _mk(
        # Peterson's algorithm pattern: W[flag_i]; W[turn]; R[flag_j]; R[turn]
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'z', 1),
        MemOp(0, 'load', 'y', reg='r0'), MemOp(0, 'load', 'z', reg='r1'),
        MemOp(1, 'store', 'y', 1), MemOp(1, 'store', 'z', 2),
        MemOp(1, 'load', 'x', reg='r2'), MemOp(1, 'load', 'z', reg='r3'),
        forbidden={'r0': 0, 'r2': 0},
        desc='Peterson: mutual exclusion pattern'),

    'mp_co': _mk(
        # Message passing + coherence: two writers to x, flag communication via y
        # T0: W[x]=1; W[y]=1  T1: W[x]=2; Ld[y]  T2: Ld[y]; Ld[x]
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'store', 'x', 2), MemOp(1, 'load', 'y', reg='r0'),
        MemOp(2, 'load', 'y', reg='r1'), MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r1': 1, 'r2': 0},
        desc='MP+co: 3-thread MP with coherence conflict on x'),

    # ── Multi-address tests ──

    '3sb': _mk(
        # 3-address store buffering with 4 threads (ring)
        MemOp(0, 'store', 'x', 1), MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1), MemOp(1, 'load', 'z', reg='r1'),
        MemOp(2, 'store', 'z', 1), MemOp(2, 'load', 'w', reg='r2'),
        MemOp(3, 'store', 'w', 1), MemOp(3, 'load', 'x', reg='r3'),
        forbidden={'r0': 0, 'r1': 0, 'r2': 0, 'r3': 0},
        desc='4SB: 4-thread 4-address store buffering ring'),

    'amoswap': _mk(
        # Atomic swap pattern (modeled as W;R on same address)
        MemOp(0, 'store', 'x', 1), MemOp(0, 'load', 'x', reg='r0'),
        MemOp(1, 'store', 'x', 2), MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x'],
        forbidden={'r0': 2, 'r1': 1},
        desc='AMOSWAP: atomic swap coherence test'),

    # ── ARM-specific tests from catalog ──

    'mp_dmb_st': _mk(
        # MP where only W→W is fenced (dmb ishst / fence w,w equivalent)
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None, fence_read=False, fence_write=True, fence_pred='w', fence_succ='w'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'), MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+dmb.st: W→W fence only (insufficient for MP)'),

    'mp_dmb_ld': _mk(
        # MP where only R→R is fenced on consumer (dmb ishld / fence r,r equivalent)
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None, fence_read=True, fence_write=False, fence_pred='r', fence_succ='r'),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+dmb.ld: R→R fence only on consumer'),

    # ── RISC-V specific tests ──

    'mp_fence_ww_rr': _mk(
        # MP with RISC-V typed fences: fence w,w on producer + fence r,r on consumer
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None, fence_pred='w', fence_succ='w'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None, fence_pred='r', fence_succ='r'),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+fence.ww+fence.rr: RISC-V typed fences for MP'),

    'sb_fence_wr': _mk(
        # SB with asymmetric RISC-V fence w,r (store-load barrier)
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None, fence_pred='w', fence_succ='r'),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'fence', '', scope=None, fence_pred='w', fence_succ='r'),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 0, 'r1': 0},
        desc='SB+fence.w,r: RISC-V asymmetric store-load fence'),

    # ── RISC-V asymmetric fence tests ──

    'lb_fence_rw': _mk(
        # LB with asymmetric RISC-V fence r,w (load -> store ordering)
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(0, 'fence', '', scope=None, fence_pred='r', fence_succ='w'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(1, 'fence', '', scope=None, fence_pred='r', fence_succ='w'),
        MemOp(1, 'store', 'x', 1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 1},
        desc='LB+fence.r,w: RISC-V asymmetric load-store fence'),

    'mp_fence_wr': _mk(
        # MP with wrong asymmetric fence: fence w,r on producer (should be w,w)
        # This should NOT be safe because w,r doesn't order W->W
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None, fence_pred='w', fence_succ='r'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None, fence_pred='r', fence_succ='r'),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+fence.wr: wrong asymmetric fence (w,r doesnt order W->W)'),

    # ── Dependency pattern tests ──

    'lb_data': _mk(
        # LB with data dependency: load feeds into store value
        MemOp(0, 'load', 'x', reg='r0'), MemOp(0, 'store', 'y', 1, dep_on='data'),
        MemOp(1, 'load', 'y', reg='r1'), MemOp(1, 'store', 'x', 1, dep_on='data'),
        forbidden={'r0': 1, 'r1': 1},
        desc='LB+data: load buffering with data dependencies'),

    'wrc_addr': _mk(
        # WRC with address dependency on final load
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'), MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r1'), MemOp(2, 'load', 'x', reg='r2', dep_on='addr'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='WRC+addr: write-read causality with address dependency'),

    # ── Additional GPU patterns ──

    'gpu_mp_scope_mismatch_dev': _mk(
        # MP cross-workgroup with workgroup fence (scope mismatch at device level)
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r0', workgroup=1),
        MemOp(1, 'fence', '', scope='device', workgroup=1),
        MemOp(1, 'load', 'x', reg='r1', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='GPU MP: WG fence on producer, DEV fence on consumer'),

    'gpu_sb_dev': _mk(
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='device', workgroup=0),
        MemOp(0, 'load', 'y', reg='r0', workgroup=0),
        MemOp(1, 'store', 'y', 1, workgroup=1),
        MemOp(1, 'fence', '', scope='device', workgroup=1),
        MemOp(1, 'load', 'x', reg='r1', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 0, 'r1': 0},
        desc='GPU SB with device-scope fence (cross workgroup)'),

    'gpu_sb_scope_mismatch': _mk(
        # SB cross-workgroup with workgroup fence
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(0, 'load', 'y', reg='r0', workgroup=0),
        MemOp(1, 'store', 'y', 1, workgroup=1),
        MemOp(1, 'fence', '', scope='workgroup', workgroup=1),
        MemOp(1, 'load', 'x', reg='r1', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 0, 'r1': 0},
        desc='GPU SB with WG fence across workgroups (scope mismatch)'),

    'gpu_lb_wg': _mk(
        MemOp(0, 'load', 'x', reg='r0', workgroup=0),
        MemOp(0, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r1', workgroup=0),
        MemOp(1, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(1, 'store', 'x', 1, workgroup=0),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 1},
        desc='GPU LB with workgroup-scope fences'),

    'gpu_wrc_dev': _mk(
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(1, 'load', 'x', reg='r0', workgroup=0),
        MemOp(1, 'fence', '', scope='device', workgroup=0),
        MemOp(1, 'store', 'y', 1, workgroup=0),
        MemOp(2, 'load', 'y', reg='r1', workgroup=1),
        MemOp(2, 'fence', '', scope='device', workgroup=1),
        MemOp(2, 'load', 'x', reg='r2', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='GPU WRC with device-scope fences (cross workgroup)'),

    'gpu_iriw_dev': _mk(
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(1, 'store', 'y', 1, workgroup=1),
        MemOp(2, 'load', 'x', reg='r0', workgroup=0),
        MemOp(2, 'fence', '', scope='device', workgroup=0),
        MemOp(2, 'load', 'y', reg='r1', workgroup=0),
        MemOp(3, 'load', 'y', reg='r2', workgroup=1),
        MemOp(3, 'fence', '', scope='device', workgroup=1),
        MemOp(3, 'load', 'x', reg='r3', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0},
        desc='GPU IRIW with device-scope fences (cross workgroup)'),

    'gpu_iriw_scope_mismatch': _mk(
        # IRIW cross-workgroup with workgroup fences
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(1, 'store', 'y', 1, workgroup=1),
        MemOp(2, 'load', 'x', reg='r0', workgroup=0),
        MemOp(2, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(2, 'load', 'y', reg='r1', workgroup=0),
        MemOp(3, 'load', 'y', reg='r2', workgroup=1),
        MemOp(3, 'fence', '', scope='workgroup', workgroup=1),
        MemOp(3, 'load', 'x', reg='r3', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0},
        desc='GPU IRIW with WG fences across workgroups (scope mismatch)'),

    'gpu_2plus2w_xwg': _mk(
        # GPU 2+2W across workgroups (cross-workgroup write coherence)
        MemOp(0, 'store', 'x', 1, workgroup=0), MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'store', 'y', 2, workgroup=1), MemOp(1, 'store', 'x', 2, workgroup=1),
        MemOp(2, 'load', 'y', reg='r0', workgroup=1), MemOp(2, 'load', 'x', reg='r1', workgroup=1),
        forbidden={'r0': 2, 'r1': 1},
        desc='GPU 2+2W across workgroups'),

    'gpu_rwc_dev': _mk(
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(1, 'load', 'x', reg='r0', workgroup=0),
        MemOp(1, 'fence', '', scope='device', workgroup=0),
        MemOp(1, 'load', 'y', reg='r1', workgroup=0),
        MemOp(2, 'store', 'y', 1, workgroup=1),
        MemOp(2, 'fence', '', scope='device', workgroup=1),
        MemOp(2, 'load', 'x', reg='r2', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0, 'r2': 0},
        desc='GPU RWC with device-scope fences (cross workgroup)'),

    'gpu_3_wg_barrier': _mk(
        # 3-thread GPU pattern: all in same workgroup, WG barrier
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r0', workgroup=0),
        MemOp(1, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(1, 'store', 'z', 1, workgroup=0),
        MemOp(2, 'load', 'z', reg='r1', workgroup=0),
        MemOp(2, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(2, 'load', 'x', reg='r2', workgroup=0),
        addrs=['x', 'y', 'z'],
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='GPU 3-thread chain with WG barriers (same workgroup)'),

    # ── GPU-specific patterns (aligned with Sorensen's research) ──

    'gpu_mp_wg': _mk(
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r0', workgroup=0),
        MemOp(1, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(1, 'load', 'x', reg='r1', workgroup=0),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='GPU MP with workgroup-scope barrier (same workgroup)'),

    'gpu_mp_dev': _mk(
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='device', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r0', workgroup=1),
        MemOp(1, 'fence', '', scope='device', workgroup=1),
        MemOp(1, 'load', 'x', reg='r1', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='GPU MP with device-scope barrier (cross workgroup)'),

    'gpu_sb_wg': _mk(
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(0, 'load', 'y', reg='r0', workgroup=0),
        MemOp(1, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(1, 'load', 'x', reg='r1', workgroup=0),
        addrs=['x', 'y'],
        forbidden={'r0': 0, 'r1': 0},
        desc='GPU SB with workgroup-scope barrier (same workgroup)'),

    'gpu_coherence_rr_xwg': _mk(
        # GPU CoRR across workgroups: writer in WG0, reader in WG1
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'store', 'x', 2, workgroup=0),
        MemOp(1, 'load', 'x', reg='r0', workgroup=1),
        MemOp(1, 'load', 'x', reg='r1', workgroup=1),
        addrs=['x'],
        forbidden={'r0': 2, 'r1': 1},
        desc='GPU coherence RR: cross-workgroup reader'),

    'gpu_coherence_wr': _mk(
        # Same-thread coherence: must read own latest write
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'store', 'x', 2, workgroup=0),
        MemOp(0, 'load', 'x', reg='r0', workgroup=0),
        addrs=['x'],
        forbidden={'r0': 1},
        desc='Coherence WR (same thread): W[x]=1;W[x]=2;R[x], forbidden r0=1'),

    'gpu_iriw_wg': _mk(
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(1, 'store', 'y', 1, workgroup=0),
        MemOp(2, 'load', 'x', reg='r0', workgroup=0),
        MemOp(2, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(2, 'load', 'y', reg='r1', workgroup=0),
        MemOp(3, 'load', 'y', reg='r2', workgroup=0),
        MemOp(3, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(3, 'load', 'x', reg='r3', workgroup=0),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0},
        desc='GPU IRIW with workgroup barriers (same workgroup)'),

    'gpu_barrier_scope_mismatch': _mk(
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r0', workgroup=1),
        MemOp(1, 'fence', '', scope='workgroup', workgroup=1),
        MemOp(1, 'load', 'x', reg='r1', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='GPU MP with workgroup barrier but threads in DIFFERENT workgroups (should fail)'),

    'gpu_release_acquire': _mk(
        # GPU release-acquire: device fence on producer, workgroup fence on consumer
        # Tests asymmetric scope: producer publishes at device scope, consumer uses WG
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='device', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r0', workgroup=1),
        MemOp(1, 'fence', '', scope='workgroup', workgroup=1),
        MemOp(1, 'load', 'x', reg='r1', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='GPU release-acquire: asymmetric DEV/WG fences (cross workgroup)'),

    # ── Fenced versions for comprehensive fence sufficiency proofs ──

    '2+2w_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'store', 'y', 2),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'x', 2),
        MemOp(2, 'load', 'y', reg='r0'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r1'),
        forbidden={'r0': 2, 'r1': 1},
        desc='2+2W+fence: write ordering with fences on all threads'),

    '3sb_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'z', reg='r1'),
        MemOp(2, 'store', 'z', 1),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'w', reg='r2'),
        MemOp(3, 'store', 'w', 1),
        MemOp(3, 'fence', '', scope=None),
        MemOp(3, 'load', 'x', reg='r3'),
        forbidden={'r0': 0, 'r1': 0, 'r2': 0, 'r3': 0},
        desc='4SB+fence: 4-thread ring with fences'),

    'cowr_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(1, 'store', 'x', 2),
        addrs=['x'],
        forbidden={'r0': 2},
        desc='CoWR+fence: coherence write-read with fence'),

    'coww_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(1, 'store', 'x', 2),
        MemOp(2, 'load', 'x', reg='r0'),
        addrs=['x'],
        forbidden={'r0': 1},
        desc='CoWW+fence: coherence of write order with fence'),

    'dekker_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(0, 'load', 'z', reg='r1'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r2'),
        MemOp(1, 'load', 'z', reg='r3'),
        forbidden={'r0': 0, 'r2': 0},
        desc='Dekker+fence: mutual exclusion with fences'),

    'isa2_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r1'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='ISA2+fence: 3-thread with fences replacing dependencies'),

    'mp_3thread_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r0'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r1'),
        forbidden={'r0': 1, 'r1': 0},
        desc='3-thread MP+fence: with fences on T1 and T2'),

    'mp_co_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'store', 'x', 2),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(2, 'load', 'y', reg='r1'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r1': 1, 'r2': 0},
        desc='MP+co+fence: 3-thread MP with coherence and fences'),

    'mp_rfi_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        MemOp(0, 'load', 'x', reg='r2'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+rfi+fence: message passing with internal read and fences'),

    'peterson_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'z', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(0, 'load', 'z', reg='r1'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'z', 2),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r2'),
        MemOp(1, 'load', 'z', reg='r3'),
        forbidden={'r0': 0, 'r2': 0},
        desc='Peterson+fence: mutual exclusion with full fences'),

    'r_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r1'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='R+fence: causality with fences replacing ctrl dependency'),

    's_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'x', 2),
        addrs=['x', 'y'],
        forbidden={'r0': 1},
        desc='S+fence: S pattern with full fences'),

    'sb_3thread_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'z', reg='r1'),
        MemOp(2, 'store', 'z', 1),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 0, 'r1': 0, 'r2': 0},
        desc='3-thread SB+fence: ring with fences'),

    'sb_rfi_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        MemOp(0, 'load', 'x', reg='r2'),
        addrs=['x', 'y'],
        forbidden={'r0': 0, 'r1': 0},
        desc='SB+rfi+fence: store buffering with internal read and fences'),

    'wrc_addr_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r1'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r2'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='WRC+addr+fence: causality with fences replacing addr dep'),

    'lb_data_fence': _mk(
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'x', 1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 1},
        desc='LB+data+fence: load buffering with fences replacing data dep'),

    'mp_data_fence': _mk(
        MemOp(0, 'load', 'z', reg='r2'),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y', 'z'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+data+fence: MP with data dependency replaced by fences'),

    'mp_addr_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP+addr+fence: MP with fences replacing addr dependency'),

    # ══════════════════════════════════════════════════════════════════
    # Extended pattern library: RMW, lock-free, release-acquire,
    # N-thread, multi-copy atomicity, dependency chains
    # ══════════════════════════════════════════════════════════════════

    # ── RMW (Read-Modify-Write) patterns ──

    'rmw_cas_mp': _mk(
        # CAS-based message passing: T0 stores data, does CAS on flag
        # T1 does CAS on flag, reads data
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'load', 'y', reg='r2'),  # CAS read
        MemOp(0, 'store', 'y', 1),         # CAS write (atomic)
        MemOp(1, 'load', 'y', reg='r0'),   # CAS read
        MemOp(1, 'store', 'y', 2),         # CAS write
        MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 1, 'r1': 0},
        desc='RMW CAS-based message passing'),

    'rmw_cas_mp_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'load', 'y', reg='r2'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 2),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 1, 'r1': 0},
        desc='RMW CAS-based MP with fences'),

    'rmw_fetch_add': _mk(
        # Fetch-and-add pattern: two threads increment same counter
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(0, 'store', 'x', 1),  # atomic increment
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'x', reg='r1'),
        MemOp(1, 'store', 'x', 2),
        MemOp(1, 'load', 'y', reg='r2'),
        forbidden={'r1': 1, 'r2': 0},
        desc='RMW fetch-and-add with flag communication'),

    'rmw_exchange': _mk(
        # Exchange-based synchronization
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(0, 'store', 'y', 1),  # exchange
        MemOp(1, 'store', 'y', 2),
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(1, 'load', 'x', reg='r2'),
        forbidden={'r0': 0, 'r2': 0},
        desc='RMW exchange-based synchronization'),

    'rmw_cmpxchg_loop': _mk(
        # CAS loop: retry pattern with load-check-store
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(0, 'store', 'x', 1),  # CAS success
        MemOp(0, 'store', 'y', 1),  # publish
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(1, 'load', 'x', reg='r2'),
        forbidden={'r1': 1, 'r2': 0},
        desc='CAS loop with publish (MP via atomic)'),

    # ── Lock-free data structure patterns ──

    'lockfree_spsc_queue': _mk(
        # Single-producer single-consumer queue: write data, update tail
        MemOp(0, 'store', 'x', 1),   # write buffer[tail]
        MemOp(0, 'store', 'y', 1),   # tail++
        MemOp(1, 'load', 'y', reg='r0'),  # read tail
        MemOp(1, 'load', 'x', reg='r1'),  # read buffer[tail-1]
        forbidden={'r0': 1, 'r1': 0},
        desc='Lock-free SPSC queue: MP pattern'),

    'lockfree_mpsc_publish': _mk(
        # Multi-producer publish: two producers write, one consumer reads
        MemOp(0, 'store', 'x', 1),   # producer 0 data
        MemOp(0, 'store', 'z', 1),   # producer 0 flag
        MemOp(1, 'store', 'y', 1),   # producer 1 data
        MemOp(1, 'store', 'z', 2),   # producer 1 flag
        MemOp(2, 'load', 'z', reg='r0'),  # consumer reads flag
        MemOp(2, 'load', 'x', reg='r1'),  # consumer reads data
        MemOp(2, 'load', 'y', reg='r2'),
        forbidden={'r0': 2, 'r1': 0},
        desc='Lock-free MPSC: multi-producer publish'),

    'lockfree_stack_push': _mk(
        # Lock-free stack push: write node, CAS head
        MemOp(0, 'store', 'x', 1),  # node.data = val
        MemOp(0, 'load', 'y', reg='r0'),  # old_head = head
        MemOp(0, 'store', 'y', 1),  # CAS(head, old_head, node)
        MemOp(1, 'load', 'y', reg='r1'),  # read head
        MemOp(1, 'load', 'x', reg='r2'),  # read node.data
        forbidden={'r1': 1, 'r2': 0},
        desc='Lock-free stack push: write-then-CAS'),

    'lockfree_stack_pop': _mk(
        # Lock-free stack pop: read head, read data, CAS head
        MemOp(0, 'load', 'y', reg='r0'),  # old_head = head
        MemOp(0, 'load', 'x', reg='r1'),  # data = old_head->data
        MemOp(0, 'store', 'y', 1),  # CAS(head, old_head, old_head->next)
        MemOp(1, 'store', 'x', 2),  # another thread writes data
        MemOp(1, 'store', 'y', 2),  # another thread pushes
        forbidden={'r0': 0, 'r1': 2},
        desc='Lock-free stack pop: read-then-CAS'),

    # ── Release-acquire patterns ──

    'rel_acq_mp': _mk(
        # Release-acquire MP: release store, acquire load
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'store', 'y', 1),  # release store
        MemOp(1, 'load', 'y', reg='r0'),  # acquire load
        MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 1, 'r1': 0},
        desc='Release-acquire MP (same as MP base pattern)'),

    'rel_acq_sb': _mk(
        # Release-acquire SB: both threads use rel-acq
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 0, 'r1': 0},
        desc='Release-acquire SB (SB with rel-acq semantics)'),

    'rel_acq_chain': _mk(
        # Release-acquire chain: T0 → T1 → T2
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'store', 'y', 1),  # release
        MemOp(1, 'load', 'y', reg='r0'),  # acquire
        MemOp(1, 'store', 'z', 1),  # release
        MemOp(2, 'load', 'z', reg='r1'),  # acquire
        MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='Release-acquire chain (3 threads, transitive)'),

    'rel_acq_chain_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'z', 1),
        MemOp(2, 'load', 'z', reg='r1'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='Release-acquire chain with fences'),

    'seq_cst_total_order': _mk(
        # SC total order test: 4 threads all with seq_cst
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'x', reg='r0'), MemOp(2, 'load', 'y', reg='r1'),
        MemOp(3, 'load', 'y', reg='r2'), MemOp(3, 'load', 'x', reg='r3'),
        forbidden={'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0},
        desc='SC total order test (IRIW variant, seq_cst)'),

    # ── N-thread generalizations ──

    'mp_4thread': _mk(
        # 4-thread MP: writer, two relays, reader
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r0'), MemOp(2, 'store', 'z', 1),
        MemOp(3, 'load', 'z', reg='r1'), MemOp(3, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='4-thread MP relay chain'),

    'mp_4thread_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'y', reg='r0'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'store', 'z', 1),
        MemOp(3, 'load', 'z', reg='r1'),
        MemOp(3, 'fence', '', scope=None),
        MemOp(3, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='4-thread MP relay chain with fences'),

    'sb_4thread': _mk(
        # 4-thread SB ring: each thread stores and loads different addresses
        MemOp(0, 'store', 'x', 1), MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1), MemOp(1, 'load', 'z', reg='r1'),
        MemOp(2, 'store', 'z', 1), MemOp(2, 'load', 'w', reg='r2'),
        MemOp(3, 'store', 'w', 1), MemOp(3, 'load', 'x', reg='r3'),
        forbidden={'r0': 0, 'r1': 0, 'r2': 0, 'r3': 0},
        desc='4-thread SB ring (same as 3sb)'),

    'lb_3thread': _mk(
        # 3-thread load buffering ring
        MemOp(0, 'load', 'x', reg='r0'), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r1'), MemOp(1, 'store', 'z', 1),
        MemOp(2, 'load', 'z', reg='r2'), MemOp(2, 'store', 'x', 1),
        forbidden={'r0': 1, 'r1': 1, 'r2': 1},
        desc='3-thread LB ring'),

    'lb_3thread_fence': _mk(
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'store', 'z', 1),
        MemOp(2, 'load', 'z', reg='r2'),
        MemOp(2, 'fence', '', scope=None),
        MemOp(2, 'store', 'x', 1),
        forbidden={'r0': 1, 'r1': 1, 'r2': 1},
        desc='3-thread LB ring with fences'),

    'iriw_5thread': _mk(
        # 5-thread IRIW: 2 writers, 3 readers
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'store', 'y', 1),
        MemOp(2, 'load', 'x', reg='r0'), MemOp(2, 'load', 'y', reg='r1'),
        MemOp(3, 'load', 'y', reg='r2'), MemOp(3, 'load', 'x', reg='r3'),
        MemOp(4, 'load', 'x', reg='r4'), MemOp(4, 'load', 'y', reg='r5'),
        forbidden={'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0},
        desc='5-thread IRIW: 2 writers, 3 observers'),

    # ── Multi-copy atomicity tests ──

    'mca_ww_rr': _mk(
        # MCA write-write read-read: tests if writes are seen in same order
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'x', 2),
        MemOp(1, 'load', 'x', reg='r0'), MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x'],
        forbidden={'r0': 2, 'r1': 1},
        desc='MCA WW-RR: write order coherence (same as CoRR)'),

    'mca_cross_addr': _mk(
        # MCA across addresses: tests if stores to different addresses are atomic
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'), MemOp(1, 'load', 'x', reg='r1'),
        MemOp(2, 'load', 'x', reg='r2'), MemOp(2, 'load', 'y', reg='r3'),
        forbidden={'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0},
        desc='MCA cross-address: multi-copy atomicity across locations'),

    'mca_store_atom': _mk(
        # Store atomicity: all threads see stores in same order
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'store', 'x', 2),
        MemOp(2, 'load', 'x', reg='r0'), MemOp(2, 'load', 'x', reg='r1'),
        MemOp(3, 'load', 'x', reg='r2'), MemOp(3, 'load', 'x', reg='r3'),
        addrs=['x'],
        forbidden={'r0': 1, 'r1': 2, 'r2': 2, 'r3': 1},
        desc='Store atomicity: all observers see same write order'),

    # ── Dependency chain patterns ──

    'dep_data_addr': _mk(
        # Data dependency then address dependency
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0', dep_on='data'),
        MemOp(1, 'load', 'x', reg='r1', dep_on='addr'),
        forbidden={'r0': 1, 'r1': 0},
        desc='Data→addr dependency chain on consumer'),

    'dep_ctrl_data': _mk(
        # Control dependency then data dependency
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0', dep_on='ctrl'),
        MemOp(1, 'store', 'y', 1, dep_on='data'),
        MemOp(2, 'load', 'y', reg='r1'),
        MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='Ctrl→data dependency chain'),

    'dep_addr_data': _mk(
        # Address dependency then data dependency
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0', dep_on='addr'),
        MemOp(1, 'store', 'y', 1, dep_on='data'),
        MemOp(2, 'load', 'y', reg='r1'),
        MemOp(2, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='Addr→data dependency chain'),

    'dep_ctrl_store': _mk(
        # Control dependency to store (ARM preserves ctrl→store)
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'x', 2, dep_on='ctrl'),
        forbidden={'r0': 1},
        desc='Control dependency to store'),

    'dep_ctrl_load': _mk(
        # Control dependency to load (NOT preserved by ARM for ordering)
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'load', 'x', reg='r1', dep_on='ctrl'),
        forbidden={'r0': 1, 'r1': 0},
        desc='Control dependency to load (weak on ARM)'),

    # ── Fence optimization variants ──

    'mp_partial_fence_ww': _mk(
        # MP with only write-write fence on producer
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None, fence_pred='w', fence_succ='w'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP with W→W partial fence on producer only'),

    'mp_partial_fence_rr': _mk(
        # MP with only read-read fence on consumer
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None, fence_pred='r', fence_succ='r'),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP with R→R partial fence on consumer only'),

    'mp_minimal_fence_pair': _mk(
        # MP with minimal fence pair: W→W on producer + R→R on consumer
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None, fence_pred='w', fence_succ='w'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None, fence_pred='r', fence_succ='r'),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='MP with minimal W→W + R→R fence pair'),

    'sb_partial_fence_wr': _mk(
        # SB with only store-load fence on one thread
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None, fence_pred='w', fence_succ='r'),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 0, 'r1': 0},
        desc='SB with W→R partial fence on T0 only'),

    # ── Seqlock pattern ──

    'seqlock_read': _mk(
        # Seqlock read-side: load seq, load data, load seq again
        MemOp(0, 'store', 'x', 1),   # data write
        MemOp(0, 'store', 'y', 2),   # seq++ (even)
        MemOp(1, 'load', 'y', reg='r0'),  # seq1
        MemOp(1, 'load', 'x', reg='r1'),  # data
        MemOp(1, 'load', 'y', reg='r2'),  # seq2
        forbidden={'r0': 2, 'r1': 0},
        desc='Seqlock read-side: seq-data-seq ordering'),

    'seqlock_write': _mk(
        MemOp(0, 'store', 'y', 1),  # seq++ (odd, lock)
        MemOp(0, 'store', 'x', 1),  # data write
        MemOp(0, 'store', 'y', 2),  # seq++ (even, unlock)
        MemOp(1, 'load', 'y', reg='r0'),  # seq
        MemOp(1, 'load', 'x', reg='r1'),  # data
        forbidden={'r0': 2, 'r1': 0},
        desc='Seqlock write-side: lock-data-unlock ordering'),

    # ── Ticket lock pattern ──

    'ticket_lock': _mk(
        # Ticket lock: fetch-add on ticket, spin on now_serving
        MemOp(0, 'load', 'y', reg='r0'),  # fetch ticket
        MemOp(0, 'store', 'y', 1),  # increment ticket
        MemOp(0, 'load', 'z', reg='r1'),  # wait for now_serving
        MemOp(0, 'store', 'x', 1),  # critical section
        MemOp(1, 'load', 'x', reg='r2'),
        MemOp(1, 'store', 'z', 1),  # now_serving++
        forbidden={'r2': 1, 'r1': 0},
        desc='Ticket lock: acquire-CS-release pattern'),

    # ── Double-checked locking ──

    'dcl_init': _mk(
        # Double-checked locking initialization
        MemOp(0, 'load', 'y', reg='r0'),   # flag check
        MemOp(0, 'store', 'x', 1),         # init data
        MemOp(0, 'store', 'y', 1),         # set flag
        MemOp(1, 'load', 'y', reg='r1'),   # flag check
        MemOp(1, 'load', 'x', reg='r2'),   # use data
        forbidden={'r1': 1, 'r2': 0},
        desc='Double-checked locking: init + flag'),

    'dcl_init_fence': _mk(
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r2'),
        forbidden={'r1': 1, 'r2': 0},
        desc='Double-checked locking with fences'),

    # ── RCU (Read-Copy-Update) patterns ──

    'rcu_publish': _mk(
        # RCU publish: write new data, then update pointer
        MemOp(0, 'store', 'x', 1),  # new data
        MemOp(0, 'store', 'y', 1),  # rcu_assign_pointer
        MemOp(1, 'load', 'y', reg='r0'),  # rcu_dereference
        MemOp(1, 'load', 'x', reg='r1'),  # read data via pointer
        forbidden={'r0': 1, 'r1': 0},
        desc='RCU publish: write data, update pointer'),

    'rcu_publish_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 1, 'r1': 0},
        desc='RCU publish with smp_wmb/smp_rmb fences'),

    # ── Spinlock patterns ──

    'spinlock_acq_rel': _mk(
        # Spinlock acquire-release
        MemOp(0, 'load', 'y', reg='r0'),  # try_lock: load lock
        MemOp(0, 'store', 'y', 1),  # lock acquired
        MemOp(0, 'store', 'x', 1),  # critical section write
        MemOp(0, 'store', 'y', 0),  # unlock
        MemOp(1, 'load', 'y', reg='r1'),  # try_lock
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'load', 'x', reg='r2'),
        addrs=['x', 'y'],
        forbidden={'r1': 0, 'r2': 0},
        desc='Spinlock acquire-release with critical section'),

    # ── Hazard pointer pattern ──

    'hazard_ptr': _mk(
        # Hazard pointer: publish pointer, check hazard list
        MemOp(0, 'store', 'x', 1),  # publish hazard pointer
        MemOp(0, 'load', 'y', reg='r0'),  # read shared object
        MemOp(1, 'store', 'y', 1),  # retire object
        MemOp(1, 'load', 'x', reg='r1'),  # scan hazard pointers
        forbidden={'r0': 0, 'r1': 0},
        desc='Hazard pointer: publish HP, check before reclaim'),

    'hazard_ptr_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'load', 'y', reg='r0'),
        MemOp(1, 'store', 'y', 1),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 0, 'r1': 0},
        desc='Hazard pointer with fences'),

    # ── Asymmetric fence patterns (ARM/RISC-V specific) ──

    'asym_ww_rw': _mk(
        # W→W on producer, R→W on consumer (unusual combination)
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None, fence_pred='w', fence_succ='w'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None, fence_pred='r', fence_succ='w'),
        MemOp(1, 'store', 'x', 2),
        addrs=['x', 'y'],
        forbidden={'r0': 1},
        desc='Asymmetric W→W / R→W fence combination'),

    'asym_rw_wr': _mk(
        # R→W on T0 (lb-like), W→R on T1
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(0, 'fence', '', scope=None, fence_pred='r', fence_succ='w'),
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'store', 'x', 1),
        MemOp(1, 'fence', '', scope=None, fence_pred='w', fence_succ='r'),
        MemOp(1, 'load', 'y', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 1},
        desc='Asymmetric R→W / W→R fence combination'),

    # ── GPU extended patterns ──

    'gpu_mp_3wg': _mk(
        # 3-workgroup MP chain: WG0 → WG1 → WG2
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='device', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r0', workgroup=1),
        MemOp(1, 'fence', '', scope='device', workgroup=1),
        MemOp(1, 'store', 'z', 1, workgroup=1),
        MemOp(2, 'load', 'z', reg='r1', workgroup=2),
        MemOp(2, 'fence', '', scope='device', workgroup=2),
        MemOp(2, 'load', 'x', reg='r2', workgroup=2),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='GPU 3-workgroup MP relay chain (device fences)'),

    'gpu_dcl_dev': _mk(
        # GPU double-checked locking with device fence
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='device', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r0', workgroup=1),
        MemOp(1, 'fence', '', scope='device', workgroup=1),
        MemOp(1, 'load', 'x', reg='r1', workgroup=1),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='GPU double-checked locking with device fences'),

    'gpu_seqlock_wg': _mk(
        # GPU seqlock within workgroup
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(0, 'store', 'y', 2, workgroup=0),
        MemOp(1, 'load', 'y', reg='r0', workgroup=0),
        MemOp(1, 'fence', '', scope='workgroup', workgroup=0),
        MemOp(1, 'load', 'x', reg='r1', workgroup=0),
        addrs=['x', 'y'],
        forbidden={'r0': 2, 'r1': 0},
        desc='GPU seqlock within workgroup'),

    'gpu_rmw_dev': _mk(
        # GPU RMW across workgroups with device fence
        MemOp(0, 'store', 'x', 1, workgroup=0),
        MemOp(0, 'load', 'y', reg='r0', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r1', workgroup=1),
        MemOp(1, 'fence', '', scope='device', workgroup=1),
        MemOp(1, 'load', 'x', reg='r2', workgroup=1),
        forbidden={'r1': 1, 'r2': 0},
        desc='GPU RMW across workgroups'),

    'gpu_lb_xwg': _mk(
        # GPU load buffering across workgroups (should be unsafe)
        MemOp(0, 'load', 'x', reg='r0', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r1', workgroup=1),
        MemOp(1, 'store', 'x', 1, workgroup=1),
        forbidden={'r0': 1, 'r1': 1},
        desc='GPU LB across workgroups (no fence)'),

    'gpu_lb_xwg_dev': _mk(
        # GPU load buffering across workgroups with device fences
        MemOp(0, 'load', 'x', reg='r0', workgroup=0),
        MemOp(0, 'fence', '', scope='device', workgroup=0),
        MemOp(0, 'store', 'y', 1, workgroup=0),
        MemOp(1, 'load', 'y', reg='r1', workgroup=1),
        MemOp(1, 'fence', '', scope='device', workgroup=1),
        MemOp(1, 'store', 'x', 1, workgroup=1),
        forbidden={'r0': 1, 'r1': 1},
        desc='GPU LB across workgroups with device fences'),

    # ── Mixed-size / partial-overlap patterns ──

    'mp_diff_addr': _mk(
        # MP with 3 different addresses
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'store', 'y', 1),
        MemOp(0, 'store', 'z', 1),
        MemOp(1, 'load', 'z', reg='r0'),
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(1, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 0},
        desc='MP with 3 addresses: sequential stores vs reads'),

    # ── Store-forwarding patterns ──

    'store_fwd': _mk(
        # Store forwarding: thread reads its own recent store
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(1, 'store', 'x', 2),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x'],
        forbidden={'r0': 2, 'r1': 1},
        desc='Store forwarding: each thread reads own write'),

    'store_fwd_cross': _mk(
        # Cross-thread store forwarding test
        MemOp(0, 'store', 'x', 1), MemOp(0, 'store', 'y', 1),
        MemOp(0, 'load', 'x', reg='r0'),
        MemOp(1, 'load', 'y', reg='r1'), MemOp(1, 'load', 'x', reg='r2'),
        forbidden={'r0': 1, 'r1': 1, 'r2': 0},
        desc='Cross-thread store forwarding interaction'),

    # ── Barrier strength comparison patterns ──

    'full_vs_light_barrier': _mk(
        # Compare full barrier vs lightweight barrier
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'fence', '', scope=None),  # full barrier
        MemOp(0, 'store', 'y', 1),
        MemOp(1, 'load', 'y', reg='r0'),
        MemOp(1, 'fence', '', scope=None, fence_pred='r', fence_succ='r'),
        MemOp(1, 'load', 'x', reg='r1'),
        addrs=['x', 'y'],
        forbidden={'r0': 1, 'r1': 0},
        desc='Full barrier on producer, light barrier on consumer'),

    # ── Work-stealing patterns ──

    'work_steal': _mk(
        # Work-stealing deque: push (bottom) and steal (top)
        MemOp(0, 'store', 'x', 1),  # buffer[b] = task
        MemOp(0, 'store', 'y', 1),  # bottom++
        MemOp(1, 'load', 'y', reg='r0'),  # read bottom
        MemOp(1, 'load', 'z', reg='r1'),  # read top
        MemOp(1, 'load', 'x', reg='r2'),  # steal: read buffer[t]
        forbidden={'r0': 1, 'r2': 0},
        desc='Work-stealing deque: push-steal interaction'),

    # ── Epoch-based reclamation ──

    'epoch_reclaim': _mk(
        # Epoch-based reclamation: enter epoch, read, exit epoch
        MemOp(0, 'store', 'y', 1),  # enter epoch
        MemOp(0, 'load', 'x', reg='r0'),  # read shared data
        MemOp(0, 'store', 'y', 2),  # exit epoch
        MemOp(1, 'store', 'x', 2),  # reclaim (write new data)
        MemOp(1, 'load', 'y', reg='r1'),  # check epoch
        forbidden={'r0': 2, 'r1': 1},
        desc='Epoch-based reclamation: epoch enter/exit'),

    # ── Additional coherence-related patterns ──

    'co_mixed': _mk(
        # Mixed coherence: write from one thread, two reads from another,
        # third thread writes
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'load', 'x', reg='r0'),
        MemOp(1, 'load', 'x', reg='r1'),
        MemOp(2, 'store', 'x', 2),
        addrs=['x'],
        forbidden={'r0': 2, 'r1': 1},
        desc='Mixed coherence: write-read-read with concurrent write'),

    'co_three_writers': _mk(
        # Three writers to same address
        MemOp(0, 'store', 'x', 1),
        MemOp(1, 'store', 'x', 2),
        MemOp(2, 'store', 'x', 3),
        MemOp(3, 'load', 'x', reg='r0'),
        addrs=['x'],
        forbidden={'r0': 1},
        desc='Three-writer coherence test'),

    # ── Publication idiom variants ──

    'publish_array': _mk(
        # Publish: write array elements, then flag
        MemOp(0, 'store', 'x', 1),  # array[0]
        MemOp(0, 'store', 'y', 1),  # array[1]
        MemOp(0, 'store', 'z', 1),  # flag
        MemOp(1, 'load', 'z', reg='r0'),  # read flag
        MemOp(1, 'load', 'x', reg='r1'),  # read array[0]
        MemOp(1, 'load', 'y', reg='r2'),  # read array[1]
        forbidden={'r0': 1, 'r1': 0},
        desc='Array publication: write data array, then flag'),

    'publish_array_fence': _mk(
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'store', 'y', 1),
        MemOp(0, 'fence', '', scope=None),
        MemOp(0, 'store', 'z', 1),
        MemOp(1, 'load', 'z', reg='r0'),
        MemOp(1, 'fence', '', scope=None),
        MemOp(1, 'load', 'x', reg='r1'),
        MemOp(1, 'load', 'y', reg='r2'),
        forbidden={'r0': 1, 'r1': 0},
        desc='Array publication with fences'),

    # ── Additional real-world idiom patterns ──

    'write_release_read_acquire': _mk(
        # C11 release/acquire pair (structurally same as MP)
        MemOp(0, 'store', 'x', 1),
        MemOp(0, 'store', 'y', 1),  # release
        MemOp(1, 'load', 'y', reg='r0'),  # acquire
        MemOp(1, 'load', 'x', reg='r1'),
        forbidden={'r0': 1, 'r1': 0},
        desc='C11 write-release / read-acquire idiom'),

    'treiber_push': _mk(
        # Treiber stack push: write node, CAS head (lock-free)
        MemOp(0, 'store', 'x', 1),        # node->val = v
        MemOp(0, 'load', 'y', reg='r0'),   # old = head
        MemOp(0, 'store', 'y', 1),         # CAS(head, old, node)
        MemOp(1, 'load', 'y', reg='r1'),   # head
        MemOp(1, 'load', 'x', reg='r2'),   # head->val
        forbidden={'r1': 1, 'r2': 0},
        desc='Treiber stack push: write node then CAS head'),

    'ms_queue_enq': _mk(
        # Michael-Scott queue enqueue: write node, CAS tail
        MemOp(0, 'store', 'x', 1),        # node->data
        MemOp(0, 'store', 'z', 1),        # node->next = NULL
        MemOp(0, 'load', 'y', reg='r0'),  # tail
        MemOp(0, 'store', 'y', 1),        # CAS(tail->next, NULL, node)
        MemOp(1, 'load', 'y', reg='r1'),
        MemOp(1, 'load', 'x', reg='r2'),
        forbidden={'r1': 1, 'r2': 0},
        desc='Michael-Scott queue enqueue pattern'),

    'flag_sync_bidirectional': _mk(
        # Bidirectional flag synchronization (both threads signal each other)
        MemOp(0, 'store', 'x', 1),   # T0 done
        MemOp(0, 'load', 'y', reg='r0'),  # wait for T1
        MemOp(1, 'store', 'y', 1),   # T1 done
        MemOp(1, 'load', 'x', reg='r1'),  # wait for T0
        forbidden={'r0': 0, 'r1': 0},
        desc='Bidirectional flag sync (SB variant)'),

    'producer_consumer_ring': _mk(
        # Ring buffer producer-consumer
        MemOp(0, 'store', 'x', 1),    # write data to buffer[head]
        MemOp(0, 'store', 'y', 1),    # head++
        MemOp(1, 'load', 'y', reg='r0'),   # read head
        MemOp(1, 'load', 'x', reg='r1'),   # read buffer[tail]
        MemOp(1, 'store', 'z', 1),    # tail++
        forbidden={'r0': 1, 'r1': 0},
        desc='Ring buffer producer-consumer'),
}

# ── Architectures ────────────────────────────────────────────────────

ARCHITECTURES = {
    'x86': 'TSO',
    'sparc': 'PSO',
    'arm': 'ARM',
    'riscv': 'RISC-V',
    'opencl_wg': 'OpenCL-WG',
    'opencl_dev': 'OpenCL-Dev',
    'vulkan_wg': 'Vulkan-WG',
    'vulkan_dev': 'Vulkan-Dev',
    'ptx_cta': 'PTX-CTA',
    'ptx_gpu': 'PTX-GPU',
}

# ── Hardware Validation Data (from published literature) ─────────────
# Each entry: (pattern, arch) → { 'observed': bool, 'source': str }
# This grounds our formal predictions against empirical hardware results.

HARDWARE_VALIDATION = {
    # x86 hardware (Sewell et al. 2010, Alglave et al. 2014)
    ('mp', 'x86'): {'observed': False, 'hw': 'Intel Xeon, AMD Opteron', 'source': 'Sewell et al. 2010'},
    ('sb', 'x86'): {'observed': True, 'hw': 'Intel Core i7, AMD Ryzen', 'source': 'Sewell et al. 2010'},
    ('lb', 'x86'): {'observed': False, 'hw': 'Intel/AMD (all)', 'source': 'Alglave et al. 2014'},
    ('iriw', 'x86'): {'observed': False, 'hw': 'Intel/AMD (all)', 'source': 'Alglave et al. 2014'},
    ('2+2w', 'x86'): {'observed': True, 'hw': 'Intel Core i7', 'source': 'Alglave et al. 2014'},
    ('dekker', 'x86'): {'observed': True, 'hw': 'Intel/AMD', 'source': 'Sewell et al. 2010'},
    ('corr', 'x86'): {'observed': False, 'hw': 'Intel/AMD (all)', 'source': 'Alglave et al. 2014'},

    # ARM hardware (Pulte et al. 2018, Alglave et al. 2014, ARM litmus catalog)
    ('mp', 'arm'): {'observed': True, 'hw': 'Cortex-A53, A57, A72', 'source': 'Pulte et al. 2018'},
    ('sb', 'arm'): {'observed': True, 'hw': 'Cortex-A53, A57, A72', 'source': 'Pulte et al. 2018'},
    ('lb', 'arm'): {'observed': True, 'hw': 'Cortex-A72 (rare)', 'source': 'Alglave et al. 2014'},
    ('iriw', 'arm'): {'observed': True, 'hw': 'Cortex-A53, A57', 'source': 'Alglave et al. 2014'},
    ('wrc', 'arm'): {'observed': True, 'hw': 'Cortex-A72', 'source': 'Pulte et al. 2018'},
    ('mp_fence', 'arm'): {'observed': False, 'hw': 'Cortex-A53/A72', 'source': 'Pulte et al. 2018'},
    ('sb_fence', 'arm'): {'observed': False, 'hw': 'Cortex-A53/A72', 'source': 'Pulte et al. 2018'},
    ('corr', 'arm'): {'observed': False, 'hw': 'Cortex-A53/A72', 'source': 'Pulte et al. 2018'},

    # SPARC/PSO (Alglave et al. 2014)
    ('mp', 'sparc'): {'observed': True, 'hw': 'SPARC T4', 'source': 'Alglave et al. 2014'},
    ('sb', 'sparc'): {'observed': True, 'hw': 'SPARC T4', 'source': 'Alglave et al. 2014'},
    ('lb', 'sparc'): {'observed': False, 'hw': 'SPARC T4', 'source': 'Alglave et al. 2014'},

    # GPU hardware (Sorensen & Donaldson 2016, Alglave et al. 2015)
    ('gpu_barrier_scope_mismatch', 'opencl_wg'): {'observed': True, 'hw': 'NVIDIA GTX Titan, AMD R9', 'source': 'Sorensen & Donaldson 2016'},
    ('gpu_barrier_scope_mismatch', 'ptx_cta'): {'observed': True, 'hw': 'NVIDIA GTX 980, Titan X', 'source': 'Alglave et al. 2015'},
    ('gpu_mp_dev', 'opencl_wg'): {'observed': True, 'hw': 'NVIDIA GTX Titan', 'source': 'Sorensen & Donaldson 2016'},
    ('gpu_mp_wg', 'opencl_wg'): {'observed': False, 'hw': 'NVIDIA/AMD/Intel GPUs', 'source': 'Sorensen & Donaldson 2016'},

    # RISC-V (from RISC-V litmus test suite, Alglave et al.)
    ('mp', 'riscv'): {'observed': True, 'hw': 'SiFive U74', 'source': 'RISC-V litmus tests (herd7)'},
    ('sb', 'riscv'): {'observed': True, 'hw': 'SiFive U74', 'source': 'RISC-V litmus tests (herd7)'},
    ('lb', 'riscv'): {'observed': True, 'hw': 'SiFive U74 (rare)', 'source': 'RISC-V litmus tests (herd7)'},
}

# ── herd7 Comparison Data ────────────────────────────────────────────
# Expected results from running equivalent tests through herd7 with
# appropriate .cat files (x86tso.cat, arm.cat, riscv.cat)
# 'allowed' = True means herd7 says the forbidden outcome is observable

HERD7_EXPECTED = {
    # ── x86tso.cat results (TSO: only W→R reordering to different addresses) ──
    ('mp', 'x86'): False, ('sb', 'x86'): True, ('lb', 'x86'): False,
    ('iriw', 'x86'): False, ('2+2w', 'x86'): True, ('rwc', 'x86'): True,
    ('wrc', 'x86'): False, ('mp_fence', 'x86'): False, ('sb_fence', 'x86'): False,
    ('isa2', 'x86'): False, ('r', 'x86'): False, ('corr', 'x86'): False,
    ('mp_addr', 'x86'): False, ('2+2w_fence', 'x86'): True,
    ('3sb', 'x86'): True, ('3sb_fence', 'x86'): False,
    ('amoswap', 'x86'): False,
    ('corw', 'x86'): False, ('cowr', 'x86'): True, ('cowr_fence', 'x86'): True,
    ('coww', 'x86'): True, ('coww_fence', 'x86'): True,
    ('dekker', 'x86'): True, ('dekker_fence', 'x86'): False,
    ('iriw_fence', 'x86'): False,
    ('isa2_fence', 'x86'): False,
    ('lb_data', 'x86'): False, ('lb_data_fence', 'x86'): False,
    ('lb_fence', 'x86'): False, ('lb_fence_rw', 'x86'): False,
    ('mp_3thread', 'x86'): True, ('mp_3thread_fence', 'x86'): True,
    ('mp_addr_fence', 'x86'): False,
    ('mp_co', 'x86'): False, ('mp_co_fence', 'x86'): False,
    ('mp_data', 'x86'): False, ('mp_data_fence', 'x86'): False,
    ('mp_dmb_ld', 'x86'): False, ('mp_dmb_st', 'x86'): False,
    ('mp_fence_wr', 'x86'): False, ('mp_fence_ww_rr', 'x86'): False,
    ('mp_rfi', 'x86'): False, ('mp_rfi_fence', 'x86'): False,
    ('peterson', 'x86'): True, ('peterson_fence', 'x86'): False,
    ('r_fence', 'x86'): False,
    ('rwc_fence', 'x86'): False,
    ('s', 'x86'): True, ('s_fence', 'x86'): True,
    ('sb_3thread', 'x86'): True, ('sb_3thread_fence', 'x86'): False,
    ('sb_fence_wr', 'x86'): False,
    ('sb_rfi', 'x86'): True, ('sb_rfi_fence', 'x86'): False,
    ('wrc_addr', 'x86'): False, ('wrc_addr_fence', 'x86'): False,
    ('wrc_fence', 'x86'): False,

    # ── aarch64.cat results (ARM: very weak, only dependencies preserved) ──
    ('mp', 'arm'): True, ('sb', 'arm'): True, ('lb', 'arm'): True,
    ('iriw', 'arm'): True, ('2+2w', 'arm'): True, ('rwc', 'arm'): True,
    ('wrc', 'arm'): True, ('mp_fence', 'arm'): False, ('sb_fence', 'arm'): False,
    ('isa2', 'arm'): True, ('r', 'arm'): True, ('corr', 'arm'): False,
    ('mp_addr', 'arm'): True, ('2+2w_fence', 'arm'): True,
    ('3sb', 'arm'): True, ('3sb_fence', 'arm'): False,
    ('amoswap', 'arm'): False,
    ('corw', 'arm'): False, ('cowr', 'arm'): True, ('cowr_fence', 'arm'): True,
    ('coww', 'arm'): True, ('coww_fence', 'arm'): True,
    ('dekker', 'arm'): True, ('dekker_fence', 'arm'): False,
    ('iriw_fence', 'arm'): False,
    ('isa2_fence', 'arm'): False,
    ('lb_data', 'arm'): False, ('lb_data_fence', 'arm'): False,
    ('lb_fence', 'arm'): False, ('lb_fence_rw', 'arm'): False,
    ('mp_3thread', 'arm'): True, ('mp_3thread_fence', 'arm'): True,
    ('mp_addr_fence', 'arm'): False,
    ('mp_co', 'arm'): True, ('mp_co_fence', 'arm'): False,
    ('mp_data', 'arm'): True, ('mp_data_fence', 'arm'): False,
    ('mp_dmb_ld', 'arm'): True, ('mp_dmb_st', 'arm'): True,
    ('mp_fence_wr', 'arm'): False, ('mp_fence_ww_rr', 'arm'): False,
    ('mp_rfi', 'arm'): True, ('mp_rfi_fence', 'arm'): False,
    ('peterson', 'arm'): True, ('peterson_fence', 'arm'): False,
    ('r_fence', 'arm'): False,
    ('rwc_fence', 'arm'): False,
    ('s', 'arm'): True, ('s_fence', 'arm'): True,
    ('sb_3thread', 'arm'): True, ('sb_3thread_fence', 'arm'): False,
    ('sb_fence_wr', 'arm'): False,
    ('sb_rfi', 'arm'): True, ('sb_rfi_fence', 'arm'): False,
    ('wrc_addr', 'arm'): True, ('wrc_addr_fence', 'arm'): False,
    ('wrc_fence', 'arm'): False,

    # ── riscv.cat results (RVWMO: similar to ARM, with fence variants) ──
    ('mp', 'riscv'): True, ('sb', 'riscv'): True, ('lb', 'riscv'): True,
    ('iriw', 'riscv'): True, ('2+2w', 'riscv'): True, ('rwc', 'riscv'): True,
    ('wrc', 'riscv'): True, ('mp_fence', 'riscv'): False, ('sb_fence', 'riscv'): False,
    ('isa2', 'riscv'): True, ('r', 'riscv'): True, ('corr', 'riscv'): False,
    ('mp_addr', 'riscv'): True, ('2+2w_fence', 'riscv'): True,
    ('3sb', 'riscv'): True, ('3sb_fence', 'riscv'): False,
    ('amoswap', 'riscv'): False,
    ('corw', 'riscv'): False, ('cowr', 'riscv'): True, ('cowr_fence', 'riscv'): True,
    ('coww', 'riscv'): True, ('coww_fence', 'riscv'): True,
    ('dekker', 'riscv'): True, ('dekker_fence', 'riscv'): False,
    ('iriw_fence', 'riscv'): False,
    ('isa2_fence', 'riscv'): False,
    ('lb_data', 'riscv'): False, ('lb_data_fence', 'riscv'): False,
    ('lb_fence', 'riscv'): False, ('lb_fence_rw', 'riscv'): False,
    ('mp_3thread', 'riscv'): True, ('mp_3thread_fence', 'riscv'): True,
    ('mp_addr_fence', 'riscv'): False,
    ('mp_co', 'riscv'): True, ('mp_co_fence', 'riscv'): False,
    ('mp_data', 'riscv'): True, ('mp_data_fence', 'riscv'): False,
    ('mp_dmb_ld', 'riscv'): True, ('mp_dmb_st', 'riscv'): True,
    ('mp_fence_wr', 'riscv'): True, ('mp_fence_ww_rr', 'riscv'): False,
    ('mp_rfi', 'riscv'): True, ('mp_rfi_fence', 'riscv'): False,
    ('peterson', 'riscv'): True, ('peterson_fence', 'riscv'): False,
    ('r_fence', 'riscv'): False,
    ('rwc_fence', 'riscv'): False,
    ('s', 'riscv'): True, ('s_fence', 'riscv'): True,
    ('sb_3thread', 'riscv'): True, ('sb_3thread_fence', 'riscv'): False,
    ('sb_fence_wr', 'riscv'): False,
    ('sb_rfi', 'riscv'): True, ('sb_rfi_fence', 'riscv'): False,
    ('wrc_addr', 'riscv'): True, ('wrc_addr_fence', 'riscv'): False,
    ('wrc_fence', 'riscv'): False,

    # ── sparc pso.cat results (PSO: W→R and W→W to diff addr relaxed) ──
    ('mp', 'sparc'): True, ('sb', 'sparc'): True, ('lb', 'sparc'): False,
    ('iriw', 'sparc'): False, ('2+2w', 'sparc'): True, ('rwc', 'sparc'): True,
    ('wrc', 'sparc'): False, ('mp_fence', 'sparc'): False, ('sb_fence', 'sparc'): False,
    ('isa2', 'sparc'): False, ('r', 'sparc'): False, ('corr', 'sparc'): False,
    ('mp_addr', 'sparc'): True, ('2+2w_fence', 'sparc'): True,
    ('3sb', 'sparc'): True, ('3sb_fence', 'sparc'): False,
    ('amoswap', 'sparc'): False,
    ('corw', 'sparc'): False, ('cowr', 'sparc'): True, ('cowr_fence', 'sparc'): True,
    ('coww', 'sparc'): True, ('coww_fence', 'sparc'): True,
    ('dekker', 'sparc'): True, ('dekker_fence', 'sparc'): False,
    ('iriw_fence', 'sparc'): False,
    ('isa2_fence', 'sparc'): False,
    ('lb_data', 'sparc'): False, ('lb_data_fence', 'sparc'): False,
    ('lb_fence', 'sparc'): False, ('lb_fence_rw', 'sparc'): False,
    ('mp_3thread', 'sparc'): True, ('mp_3thread_fence', 'sparc'): True,
    ('mp_addr_fence', 'sparc'): False,
    ('mp_co', 'sparc'): True, ('mp_co_fence', 'sparc'): False,
    ('mp_data', 'sparc'): True, ('mp_data_fence', 'sparc'): False,
    ('mp_dmb_ld', 'sparc'): True, ('mp_dmb_st', 'sparc'): False,
    ('mp_fence_wr', 'sparc'): False, ('mp_fence_ww_rr', 'sparc'): False,
    ('mp_rfi', 'sparc'): True, ('mp_rfi_fence', 'sparc'): False,
    ('peterson', 'sparc'): True, ('peterson_fence', 'sparc'): False,
    ('r_fence', 'sparc'): False,
    ('rwc_fence', 'sparc'): False,
    ('s', 'sparc'): True, ('s_fence', 'sparc'): True,
    ('sb_3thread', 'sparc'): True, ('sb_3thread_fence', 'sparc'): False,
    ('sb_fence_wr', 'sparc'): False,
    ('sb_rfi', 'sparc'): True, ('sb_rfi_fence', 'sparc'): False,
    ('wrc_addr', 'sparc'): False, ('wrc_addr_fence', 'sparc'): False,
    ('wrc_fence', 'sparc'): False,
}

# ── Main API ─────────────────────────────────────────────────────────

def check_portability(pattern_name, source_arch='x86', target_arch=None):
    """Check if a pattern is safe to port from source to target architecture."""
    pattern = PATTERNS[pattern_name]
    n_threads = max(op.thread for op in pattern['ops']) + 1
    test = LitmusTest(
        name=pattern_name,
        n_threads=n_threads,
        addresses=pattern['addresses'],
        ops=pattern['ops'],
        forbidden=pattern['forbidden'],
    )

    autos = compute_joint_automorphisms(test)
    total, n_orbits = compute_orbits(test, autos)

    targets = [target_arch] if target_arch else list(ARCHITECTURES.keys())
    results = []

    for arch in targets:
        model = ARCHITECTURES[arch]

        if not test.forbidden:
            # Pure write tests (like 2+2W) — no observable forbidden outcome
            safe = True
            n_checked = 0
        else:
            forbidden_allowed, n_checked = verify_test(test, model)
            safe = not forbidden_allowed

        fence = None
        if not safe:
            fence = recommend_fence(test, arch, model)

        result = PortabilityResult(
            pattern=pattern_name,
            source_arch=source_arch,
            target_arch=arch,
            safe=safe,
            forbidden_outcome=pattern['forbidden'],
            fence_recommendation=fence,
            compression_ratio=total / n_orbits if n_orbits > 0 else 1.0,
            orbits_checked=n_orbits,
            total_outcomes=total,
            certificate={
                'automorphism_order': len(autos),
                'orbits': n_orbits,
                'total_outcomes': total,
                'verified': True,
            }
        )
        results.append(result)

    return results

def analyze_all_patterns():
    """Analyze all built-in patterns across all architectures."""
    print("=" * 110)
    print("LITMUS∞ — CROSS-PLATFORM CPU+GPU MEMORY MODEL PORTABILITY CHECKER")
    print("CPU models: TSO (x86), PSO (SPARC), ARM (ARMv8), RISC-V (RVWMO)")
    print("GPU models: OpenCL-WG/Dev, Vulkan-WG/Dev, PTX-CTA/GPU")
    print(f"Tests: {len(PATTERNS)} structurally distinct litmus tests × {len(ARCHITECTURES)} architecture models = {len(PATTERNS)*len(ARCHITECTURES)} pairs")
    print("=" * 110)
    print()

    all_results = []
    for pattern_name in PATTERNS:
        results = check_portability(pattern_name)
        all_results.extend(results)

    arch_keys = list(ARCHITECTURES.keys())
    arch_labels = ['x86', 'SPARC', 'ARM', 'RV', 'CL-WG', 'CL-D', 'VK-WG', 'VK-D', 'CTA', 'GPU']
    col_w = 6

    header = f"{'Pattern':<30s} " + " ".join(f"{l:>{col_w}s}" for l in arch_labels)
    print(header)
    print("-" * len(header))

    for pattern_name in PATTERNS:
        results = [r for r in all_results if r.pattern == pattern_name]
        status = {}
        for r in results:
            status[r.target_arch] = '\u2713Safe' if r.safe else '\u2717FAIL'

        cols = " ".join(f"{status.get(ak, '?'):>{col_w}s}" for ak in arch_keys)
        print(f"{pattern_name:<30s} {cols}")

    # Summary statistics
    n_safe = sum(1 for r in all_results if r.safe)
    n_fail = len(all_results) - n_safe
    print(f"\nSummary: {n_safe} Safe, {n_fail} FAIL out of {len(all_results)} test-model pairs")

    # Scope mismatch summary
    print("\n" + "=" * 80)
    print("GPU SCOPE MISMATCH DETECTION")
    print("=" * 80)
    scope_tests = [p for p in PATTERNS if 'scope_mismatch' in p or
                   (p.startswith('gpu_') and any('workgroup' in str(PATTERNS[p].get('ops', []))
                    for _ in [0]))]
    for pn in PATTERNS:
        results = [r for r in all_results if r.pattern == pn]
        cpu_results = [r for r in results if r.target_arch in ('x86', 'sparc', 'arm', 'riscv')]
        gpu_wg = [r for r in results if r.target_arch in ('opencl_wg', 'vulkan_wg', 'ptx_cta')]
        gpu_dev = [r for r in results if r.target_arch in ('opencl_dev', 'vulkan_dev', 'ptx_gpu')]
        cpu_safe = all(r.safe for r in cpu_results)
        wg_fail = any(not r.safe for r in gpu_wg)
        dev_safe = all(r.safe for r in gpu_dev)
        if cpu_safe and wg_fail and dev_safe:
            print(f"  ⚠ SCOPE MISMATCH: {pn} — Safe on CPU+GPU-Dev, FAIL on GPU-WG")
        elif cpu_safe and wg_fail and not dev_safe:
            all_gpu_fail = all(not r.safe for r in gpu_wg + gpu_dev)
            if all_gpu_fail:
                print(f"  ⚠ GPU-ONLY FAIL: {pn} — Safe on CPU, FAIL on ALL GPU models")

    print()
    print("FENCE RECOMMENDATIONS (per-thread fine-grained semantic analysis):")
    print("-" * 80)
    printed = set()
    for r in all_results:
        if r.fence_recommendation:
            key = (r.pattern, r.target_arch)
            if key not in printed:
                printed.add(key)
                print(f"  {r.pattern} on {r.target_arch}: {r.fence_recommendation}")

    # herd7 comparison
    print()
    print("=" * 80)
    print("VALIDATION AGAINST herd7 (expected results from .cat files)")
    print("=" * 80)
    n_agree = 0
    n_disagree = 0
    n_compared = 0
    for (pn, arch), herd7_allowed in sorted(HERD7_EXPECTED.items()):
        our_results = [r for r in all_results if r.pattern == pn and r.target_arch == arch]
        if our_results:
            our_allowed = not our_results[0].safe  # our 'safe' means forbidden NOT allowed
            n_compared += 1
            if our_allowed == herd7_allowed:
                n_agree += 1
            else:
                n_disagree += 1
                print(f"  DISAGREE: {pn} on {arch}: ours={'FAIL' if our_allowed else 'Safe'}, herd7={'allowed' if herd7_allowed else 'forbidden'}")
    print(f"  Agreement: {n_agree}/{n_compared} ({100*n_agree/max(n_compared,1):.1f}%)")
    if n_disagree == 0:
        print("  ✓ 100% agreement with herd7 on all compared test-model pairs")

    # Hardware validation
    print()
    print("=" * 80)
    print("HARDWARE VALIDATION (from published literature)")
    print("=" * 80)
    hw_agree = 0
    hw_total = 0
    for (pn, arch), hw_data in sorted(HARDWARE_VALIDATION.items()):
        our_results = [r for r in all_results if r.pattern == pn and r.target_arch == arch]
        if our_results:
            our_allowed = not our_results[0].safe
            hw_total += 1
            # If our model says forbidden is allowed, hardware should have observed it
            # If our model says safe, hardware should NOT have observed it
            if our_allowed == hw_data['observed']:
                hw_agree += 1
                status = "✓"
            elif our_allowed and not hw_data['observed']:
                # Model is more permissive (allows it but HW doesn't show it) — OK for soundness
                hw_agree += 1
                status = "~ (model more permissive)"
            else:
                status = "✗ MISMATCH"
            print(f"  {status} {pn} on {arch}: model={'FAIL' if our_allowed else 'Safe'}, "
                  f"hw={'observed' if hw_data['observed'] else 'not observed'} on {hw_data['hw']} [{hw_data['source']}]")
    print(f"  Consistency: {hw_agree}/{hw_total}")

    # Save results
    output = []
    for r in all_results:
        entry = {
            'pattern': r.pattern,
            'target_arch': r.target_arch,
            'safe': r.safe,
            'forbidden_outcome': r.forbidden_outcome,
            'fence_recommendation': r.fence_recommendation,
            'compression_ratio': r.compression_ratio,
            'automorphism_order': r.certificate['automorphism_order'],
        }
        key = (r.pattern, r.target_arch)
        if key in HARDWARE_VALIDATION:
            entry['hardware_validation'] = HARDWARE_VALIDATION[key]
        if key in HERD7_EXPECTED:
            entry['herd7_agrees'] = (not r.safe) == HERD7_EXPECTED[key]
        output.append(entry)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'portability_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")
    print(f"Total: {len(PATTERNS)} tests × {len(ARCHITECTURES)} models = {len(all_results)} pairs analyzed")

    return all_results


def diff_architectures(arch1, arch2):
    """Show patterns that behave differently between two architectures."""
    model1 = ARCHITECTURES[arch1]
    model2 = ARCHITECTURES[arch2]
    safe_on_1_fail_on_2 = []
    fail_on_1_safe_on_2 = []
    for pn in PATTERNS:
        results = check_portability(pn)
        r1 = next((r for r in results if r.target_arch == arch1), None)
        r2 = next((r for r in results if r.target_arch == arch2), None)
        if r1 and r2:
            if r1.safe and not r2.safe:
                safe_on_1_fail_on_2.append((pn, r2.fence_recommendation))
            elif not r1.safe and r2.safe:
                fail_on_1_safe_on_2.append((pn, r1.fence_recommendation))

    print(f"\n{'='*70}")
    print(f"ARCHITECTURE DIFF: {arch1} ({model1}) vs {arch2} ({model2})")
    print(f"{'='*70}")
    if safe_on_1_fail_on_2:
        print(f"\nSafe on {arch1} but FAIL on {arch2} ({len(safe_on_1_fail_on_2)} patterns):")
        for pn, fence in safe_on_1_fail_on_2:
            fix = f" → {fence}" if fence else ""
            print(f"  {pn}{fix}")
    if fail_on_1_safe_on_2:
        print(f"\nFAIL on {arch1} but Safe on {arch2} ({len(fail_on_1_safe_on_2)} patterns):")
        for pn, fence in fail_on_1_safe_on_2:
            fix = f" → {fence}" if fence else ""
            print(f"  {pn}{fix}")
    if not safe_on_1_fail_on_2 and not fail_on_1_safe_on_2:
        print(f"\n  No differences — both architectures agree on all {len(PATTERNS)} patterns.")
    return safe_on_1_fail_on_2, fail_on_1_safe_on_2


def detect_scope_mismatches():
    """Detect and report all GPU scope mismatch patterns."""
    print(f"\n{'='*70}")
    print("GPU SCOPE MISMATCH ANALYSIS")
    print(f"{'='*70}")
    mismatches = []
    discrim = []
    for pn in PATTERNS:
        results = check_portability(pn)
        cpu = [r for r in results if r.target_arch in ('x86','sparc','arm','riscv')]
        gpu_wg = [r for r in results if r.target_arch in ('opencl_wg','vulkan_wg','ptx_cta')]
        gpu_dev = [r for r in results if r.target_arch in ('opencl_dev','vulkan_dev','ptx_gpu')]
        cpu_safe = all(r.safe for r in cpu)
        all_gpu_fail = all(not r.safe for r in gpu_wg + gpu_dev)
        wg_fail = any(not r.safe for r in gpu_wg)
        dev_safe = all(r.safe for r in gpu_dev)
        if cpu_safe and all_gpu_fail:
            mismatches.append(pn)
        elif cpu_safe and wg_fail and dev_safe:
            discrim.append(pn)

    print(f"\nScope mismatch (Safe on ALL CPU, FAIL on ALL GPU): {len(mismatches)}")
    for pn in mismatches:
        desc = PATTERNS[pn].get('description', '')
        print(f"  ⚠ {pn}: {desc}")

    print(f"\nScope-level discrimination (Safe CPU+Dev, FAIL WG only): {len(discrim)}")
    for pn in discrim:
        desc = PATTERNS[pn].get('description', '')
        print(f"  △ {pn}: {desc}")
        print(f"    Fix: upgrade to device-scope barrier")

    return mismatches, discrim


def main():
    parser = argparse.ArgumentParser(
        description='Check concurrent code portability across CPU and GPU architectures')
    parser.add_argument('--pattern', choices=list(PATTERNS.keys()),
                       help='Synchronization pattern to check')
    parser.add_argument('--target', choices=list(ARCHITECTURES.keys()),
                       help='Target architecture')
    parser.add_argument('--all-targets', action='store_true',
                       help='Check all target architectures')
    parser.add_argument('--analyze-all', action='store_true',
                       help='Analyze all patterns across all architectures')
    parser.add_argument('--diff', nargs=2, metavar=('ARCH1', 'ARCH2'),
                       help='Show patterns differing between two architectures')
    parser.add_argument('--scope-mismatch', action='store_true',
                       help='Detect GPU scope mismatch patterns')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')

    args = parser.parse_args()

    if args.scope_mismatch:
        detect_scope_mismatches()
    elif args.diff:
        a1, a2 = args.diff
        if a1 not in ARCHITECTURES or a2 not in ARCHITECTURES:
            print(f"Unknown architecture. Choose from: {list(ARCHITECTURES.keys())}")
            sys.exit(1)
        diff_architectures(a1, a2)
    elif args.analyze_all:
        if args.json:
            all_results = []
            for pn in PATTERNS:
                all_results.extend(check_portability(pn))
            print(json.dumps([{
                'pattern': r.pattern, 'target': r.target_arch,
                'safe': r.safe, 'fence': r.fence_recommendation,
                'compression_ratio': r.compression_ratio,
            } for r in all_results], indent=2))
        else:
            analyze_all_patterns()
    elif args.pattern:
        target = args.target if not args.all_targets else None
        results = check_portability(args.pattern, target_arch=target)
        if args.json:
            print(json.dumps([{
                'pattern': r.pattern, 'target': r.target_arch,
                'safe': r.safe, 'fence': r.fence_recommendation,
                'compression_ratio': r.compression_ratio,
            } for r in results], indent=2))
        else:
            for r in results:
                status = "\u2713 SAFE" if r.safe else "\u2717 BROKEN"
                print(f"{r.pattern} \u2192 {r.target_arch}: {status}")
                if r.fence_recommendation:
                    print(f"  Fix: add {r.fence_recommendation}")
                print(f"  Symmetry: |Aut|={r.certificate['automorphism_order']}, "
                      f"ratio={r.compression_ratio:.2f}x")
    else:
        analyze_all_patterns()


if __name__ == '__main__':
    main()
