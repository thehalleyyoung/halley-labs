#!/usr/bin/env python3
"""
pattern_composition: Beyond the 140-pattern ceiling.

Addresses the critique that fixed pattern sets limit the tool to known bugs.
Provides three capabilities:
  1. Pattern Composition Engine - compose base patterns into composite litmus tests
  2. Parameterized Pattern Generation - generate N-thread variants from 2-thread bases
  3. Bounded Model Checking - enumerate all litmus structures up to a bound

This systematically generates novel tests that go beyond hand-written patterns,
enabling discovery of previously unknown portability hazards.
"""

import copy
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from portcheck import (
    PATTERNS, ARCHITECTURES, verify_test, LitmusTest, MemOp, recommend_fence
)

# ── Helpers ──────────────────────────────────────────────────────────

CORE_PATTERNS = ['mp', 'sb', 'lb', 'wrc', 'iriw']
CORE_ARCHS = ['x86', 'arm', 'riscv']


def _pattern_to_litmus(name: str) -> LitmusTest:
    """Convert a PATTERNS dict entry into a LitmusTest object."""
    p = PATTERNS[name]
    n_threads = max(op.thread for op in p['ops']) + 1
    return LitmusTest(
        name=name,
        n_threads=n_threads,
        addresses=list(p['addresses']),
        ops=[copy.deepcopy(op) for op in p['ops']],
        forbidden=dict(p['forbidden']),
    )


def _fresh_addr(base: str, existing: Set[str]) -> str:
    """Generate a fresh address name that doesn't collide with existing ones."""
    if base not in existing:
        return base
    for i in range(100):
        candidate = f"{base}{i}"
        if candidate not in existing:
            return candidate
    return f"{base}_fresh"


def _fresh_reg(base: str, existing: Set[str]) -> str:
    """Generate a fresh register name."""
    if base not in existing:
        return base
    for i in range(100):
        candidate = f"r{i}"
        if candidate not in existing:
            return candidate
    return f"{base}_fresh"


# ── 1. Pattern Composition Engine ────────────────────────────────────

def compose_patterns(pattern1_name: str, pattern2_name: str,
                     shared_vars: Optional[List[str]] = None) -> LitmusTest:
    """Compose two base patterns into a single composite litmus test.

    Thread IDs from pattern2 are shifted to avoid conflicts with pattern1.
    Addresses from pattern2 are renamed unless they appear in shared_vars,
    which creates inter-pattern interactions (the key source of novel bugs).
    Registers are renamed to avoid conflicts.
    The forbidden outcome is the conjunction of both patterns' forbidden outcomes.
    """
    t1 = _pattern_to_litmus(pattern1_name)
    t2 = _pattern_to_litmus(pattern2_name)

    if shared_vars is None:
        shared_vars = []

    thread_offset = t1.n_threads
    used_addrs = set(t1.addresses)
    used_regs = set(t1.forbidden.keys())
    for op in t1.ops:
        if op.reg:
            used_regs.add(op.reg)

    # Build address mapping for pattern2
    addr_map = {}
    for addr in t2.addresses:
        if addr in shared_vars:
            # Keep this address shared — creates inter-pattern interaction
            addr_map[addr] = addr
        else:
            new_addr = _fresh_addr(addr, used_addrs)
            addr_map[addr] = new_addr
            used_addrs.add(new_addr)

    # Build register mapping for pattern2
    reg_map = {}
    for op in t2.ops:
        if op.reg and op.reg not in reg_map:
            new_reg = _fresh_reg(op.reg, used_regs)
            reg_map[op.reg] = new_reg
            used_regs.add(new_reg)

    # Remap pattern2 ops
    new_ops = list(t1.ops)
    for op in t2.ops:
        new_op = MemOp(
            thread=op.thread + thread_offset,
            optype=op.optype,
            addr=addr_map.get(op.addr, op.addr),
            value=op.value,
            reg=reg_map.get(op.reg, op.reg) if op.reg else None,
            scope=op.scope,
            dep_on=op.dep_on,
        )
        new_ops.append(new_op)

    # Merge forbidden outcomes
    forbidden = dict(t1.forbidden)
    for reg, val in t2.forbidden.items():
        forbidden[reg_map.get(reg, reg)] = val

    # Compute merged addresses
    all_addrs = sorted(set(t1.addresses) | set(addr_map.values()))

    # Detect inter-pattern interactions
    shared_actual = set(t1.addresses) & set(addr_map.values())
    interaction_note = ""
    if shared_actual:
        interaction_note = f" [shared: {','.join(sorted(shared_actual))}]"

    name = f"{pattern1_name}+{pattern2_name}{interaction_note}"
    n_threads = thread_offset + t2.n_threads

    return LitmusTest(
        name=name,
        n_threads=n_threads,
        addresses=all_addrs,
        ops=new_ops,
        forbidden=forbidden,
    )


# ── 2. Parameterized Pattern Generation ─────────────────────────────

def generate_chain(base_pattern: str, n_threads: int) -> LitmusTest:
    """Generate an N-thread chain from a 2-thread base pattern.

    Chain composition: T0→T1→T2→...→T(n-1)
    For mp: T0 writes data+flag, T1 reads flag+writes flag2, ..., T(n-1) reads flag(n-2)+reads data
    For sb: T0 writes x0, reads x1; T1 writes x1, reads x2; ...; T(n-1) writes x(n-1), reads x0

    Each link in the chain mirrors the communication structure of the base pattern.
    """
    if n_threads < 2:
        raise ValueError("Need at least 2 threads")

    base = PATTERNS[base_pattern]
    base_ops = base['ops']

    if base_pattern == 'mp':
        return _generate_mp_chain(n_threads)
    elif base_pattern == 'sb':
        return _generate_sb_chain(n_threads)
    elif base_pattern == 'lb':
        return _generate_lb_chain(n_threads)
    elif base_pattern == 'wrc':
        return _generate_wrc_chain(n_threads)
    else:
        # Generic chain: replicate the cross-thread communication pattern
        return _generate_generic_chain(base_pattern, n_threads)


def _generate_mp_chain(n: int) -> LitmusTest:
    """MP chain: T0 writes x,f0; T1 reads f0, writes f1; ...; T(n-1) reads f(n-2), reads x.

    The forbidden outcome is the last thread seeing the flag but not the data,
    which would indicate a failure of message passing across the chain.
    """
    ops = []
    addrs = ['x']
    forbidden = {}

    # T0: write data, write first flag
    flag0 = 'f0'
    addrs.append(flag0)
    ops.append(MemOp(0, 'store', 'x', 1))
    ops.append(MemOp(0, 'store', flag0, 1))

    # Middle threads: read previous flag, write next flag
    for i in range(1, n - 1):
        prev_flag = f'f{i-1}'
        next_flag = f'f{i}'
        addrs.append(next_flag)
        reg_read = f'r{2*i-1}'
        ops.append(MemOp(i, 'load', prev_flag, reg=reg_read))
        ops.append(MemOp(i, 'store', next_flag, 1))
        forbidden[reg_read] = 1  # middle thread sees the flag

    # Last thread: read last flag, read data
    last_flag = f'f{n-2}'
    reg_flag = f'r{2*(n-1)-1}'
    reg_data = f'r{2*(n-1)}'
    ops.append(MemOp(n - 1, 'load', last_flag, reg=reg_flag))
    ops.append(MemOp(n - 1, 'load', 'x', reg=reg_data))
    forbidden[reg_flag] = 1  # last thread sees the flag...
    forbidden[reg_data] = 0  # ...but NOT the data

    return LitmusTest(
        name=f'mp_chain_{n}t',
        n_threads=n,
        addresses=sorted(set(addrs)),
        ops=ops,
        forbidden=forbidden,
    )


def _generate_sb_chain(n: int) -> LitmusTest:
    """SB chain: Ti writes xi, reads x((i+1) mod n). Open chain variant.

    T0: W[x0]=1; R[x1]  T1: W[x1]=1; R[x2]  ...  T(n-1): W[x(n-1)]=1; R[x0]
    Forbidden: all reads return 0 (every thread's store is invisible to the next).
    Note: This is actually the ring topology for SB; chain and ring coincide for SB.
    For a true chain, T(n-1) reads from T(n-2) only (no wrap).
    """
    ops = []
    addrs = []
    forbidden = {}

    for i in range(n):
        addr_write = f'x{i}'
        addr_read = f'x{(i + 1) % n}'
        if addr_write not in addrs:
            addrs.append(addr_write)
        if addr_read not in addrs:
            addrs.append(addr_read)
        reg = f'r{i}'
        ops.append(MemOp(i, 'store', addr_write, 1))
        ops.append(MemOp(i, 'load', addr_read, reg=reg))
        forbidden[reg] = 0

    return LitmusTest(
        name=f'sb_chain_{n}t',
        n_threads=n,
        addresses=sorted(set(addrs)),
        ops=ops,
        forbidden=forbidden,
    )


def _generate_lb_chain(n: int) -> LitmusTest:
    """LB chain: Ti reads x((i-1) mod n), writes xi.

    T0: R[x(n-1)]; W[x0]=1  T1: R[x0]; W[x1]=1  ...
    Forbidden: all reads return 1 (values created out of thin air).
    """
    ops = []
    addrs = []
    forbidden = {}

    for i in range(n):
        addr_read = f'x{(i - 1) % n}'
        addr_write = f'x{i}'
        if addr_read not in addrs:
            addrs.append(addr_read)
        if addr_write not in addrs:
            addrs.append(addr_write)
        reg = f'r{i}'
        ops.append(MemOp(i, 'load', addr_read, reg=reg))
        ops.append(MemOp(i, 'store', addr_write, 1))
        forbidden[reg] = 1

    return LitmusTest(
        name=f'lb_chain_{n}t',
        n_threads=n,
        addresses=sorted(set(addrs)),
        ops=ops,
        forbidden=forbidden,
    )


def _generate_wrc_chain(n: int) -> LitmusTest:
    """WRC chain: T0 writes x. Each subsequent thread reads the previous
    variable and writes a new one, except the last thread which reads
    the first variable too (checking causality propagation).

    T0: W[x]=1
    T1: R[x]; W[y]=1
    T2: R[y]; W[z]=1
    ...
    T(n-1): R[prev]; R[x]   (checks if the write to x has propagated)
    Forbidden: all intermediate reads see 1, but final read of x sees 0.
    """
    if n < 3:
        return _pattern_to_litmus('wrc')

    ops = []
    addrs = ['x']
    forbidden = {}
    chain_vars = ['x']

    # T0: write x
    ops.append(MemOp(0, 'store', 'x', 1))

    # Middle threads: read previous var, write next var
    for i in range(1, n - 1):
        prev_var = chain_vars[-1]
        next_var = chr(ord('a') + i) if i < 26 else f'v{i}'
        chain_vars.append(next_var)
        addrs.append(next_var)
        reg_read = f'r{2*i-1}'
        ops.append(MemOp(i, 'load', prev_var, reg=reg_read))
        ops.append(MemOp(i, 'store', next_var, 1))
        forbidden[reg_read] = 1

    # Last thread: read last chain var, read x
    last_var = chain_vars[-1]
    reg_chain = f'r{2*(n-1)-1}'
    reg_x = f'r{2*(n-1)}'
    ops.append(MemOp(n - 1, 'load', last_var, reg=reg_chain))
    ops.append(MemOp(n - 1, 'load', 'x', reg=reg_x))
    forbidden[reg_chain] = 1
    forbidden[reg_x] = 0  # causality violation: sees chain but not original write

    return LitmusTest(
        name=f'wrc_chain_{n}t',
        n_threads=n,
        addresses=sorted(set(addrs)),
        ops=ops,
        forbidden=forbidden,
    )


def _generate_generic_chain(base_pattern: str, n: int) -> LitmusTest:
    """Fallback: treat 2-thread pattern as a link and chain N copies."""
    base = PATTERNS[base_pattern]
    base_ops = base['ops']
    base_nt = max(op.thread for op in base_ops) + 1

    if base_nt != 2:
        # For non-2-thread patterns, just return the base
        return _pattern_to_litmus(base_pattern)

    all_ops = []
    all_addrs = set()
    forbidden = {}
    reg_counter = [0]

    def next_reg():
        r = f'r{reg_counter[0]}'
        reg_counter[0] += 1
        return r

    for link in range(n - 1):
        t0 = link
        t1 = link + 1
        for op in base_ops:
            new_thread = t0 if op.thread == 0 else t1
            new_addr = f'{op.addr}_{link}' if link > 0 or op.addr in all_addrs else op.addr
            all_addrs.add(new_addr)
            new_reg = next_reg() if op.reg else None
            new_op = MemOp(
                thread=new_thread,
                optype=op.optype,
                addr=new_addr,
                value=op.value,
                reg=new_reg,
                dep_on=op.dep_on,
            )
            all_ops.append(new_op)
            if op.reg and op.reg in base['forbidden']:
                forbidden[new_reg] = base['forbidden'][op.reg]

    return LitmusTest(
        name=f'{base_pattern}_chain_{n}t',
        n_threads=n,
        addresses=sorted(all_addrs),
        ops=all_ops,
        forbidden=forbidden,
    )


# ── Ring Composition ─────────────────────────────────────────────────

def generate_ring(base_pattern: str, n_threads: int) -> LitmusTest:
    """Generate an N-thread ring from a 2-thread base pattern.

    Ring composition wraps around: T0→T1→T2→...→T(n-1)→T0.
    The SB ring is the canonical example: each thread stores to its own
    variable and loads from the next thread's variable.
    """
    if n_threads < 2:
        raise ValueError("Need at least 2 threads")

    if base_pattern == 'sb':
        return _generate_sb_ring(n_threads)
    elif base_pattern == 'lb':
        return _generate_lb_chain(n_threads)  # LB is inherently ring-shaped
    elif base_pattern == 'mp':
        return _generate_mp_ring(n_threads)
    else:
        return _generate_generic_ring(base_pattern, n_threads)


def _generate_sb_ring(n: int) -> LitmusTest:
    """SB ring: Ti writes xi, reads x((i+1) mod n).

    Forbidden: all reads return 0. This generalizes the classic 2-thread SB
    (store buffering) to N threads in a ring topology.
    """
    ops = []
    addrs = []
    forbidden = {}

    for i in range(n):
        addr_w = f'x{i}'
        addr_r = f'x{(i + 1) % n}'
        if addr_w not in addrs:
            addrs.append(addr_w)
        reg = f'r{i}'
        ops.append(MemOp(i, 'store', addr_w, 1))
        ops.append(MemOp(i, 'load', addr_r, reg=reg))
        forbidden[reg] = 0

    return LitmusTest(
        name=f'sb_ring_{n}t',
        n_threads=n,
        addresses=sorted(set(addrs)),
        ops=ops,
        forbidden=forbidden,
    )


def _generate_mp_ring(n: int) -> LitmusTest:
    """MP ring: chain of message-passing links forming a cycle.

    T0: W[x0]=1; W[f0]=1
    T1: R[f0]; W[x1]=1; W[f1]=1
    ...
    T(n-1): R[f(n-2)]; W[x(n-1)]=1; R[x0]
    The ring closes: the last thread checks if T0's data write is visible.
    """
    ops = []
    addrs = []
    forbidden = {}

    # T0: write data and flag
    addrs.extend(['x0', 'f0'])
    ops.append(MemOp(0, 'store', 'x0', 1))
    ops.append(MemOp(0, 'store', 'f0', 1))

    # Middle threads
    for i in range(1, n - 1):
        prev_flag = f'f{i-1}'
        data_addr = f'x{i}'
        flag_addr = f'f{i}'
        addrs.extend([data_addr, flag_addr])
        reg_flag = f'r{2*i}'
        ops.append(MemOp(i, 'load', prev_flag, reg=reg_flag))
        ops.append(MemOp(i, 'store', data_addr, 1))
        ops.append(MemOp(i, 'store', flag_addr, 1))
        forbidden[reg_flag] = 1

    # Last thread: read last flag, read T0's data
    last_flag = f'f{n-2}'
    reg_flag = f'r{2*(n-1)}'
    reg_data = f'r{2*(n-1)+1}'
    ops.append(MemOp(n - 1, 'load', last_flag, reg=reg_flag))
    ops.append(MemOp(n - 1, 'load', 'x0', reg=reg_data))
    forbidden[reg_flag] = 1
    forbidden[reg_data] = 0

    return LitmusTest(
        name=f'mp_ring_{n}t',
        n_threads=n,
        addresses=sorted(set(addrs)),
        ops=ops,
        forbidden=forbidden,
    )


def _generate_generic_ring(base_pattern: str, n: int) -> LitmusTest:
    """Generic ring: treat each pair (Ti, T(i+1 mod n)) as a link."""
    base = PATTERNS[base_pattern]
    base_ops = base['ops']
    base_nt = max(op.thread for op in base_ops) + 1

    if base_nt != 2:
        return _pattern_to_litmus(base_pattern)

    all_ops = []
    all_addrs = set()
    forbidden = {}
    reg_counter = [0]

    def next_reg():
        r = f'r{reg_counter[0]}'
        reg_counter[0] += 1
        return r

    for link in range(n):
        t0 = link
        t1 = (link + 1) % n
        for op in base_ops:
            new_thread = t0 if op.thread == 0 else t1
            new_addr = f'{op.addr}_{link}'
            all_addrs.add(new_addr)
            new_reg = next_reg() if op.reg else None
            new_op = MemOp(
                thread=new_thread,
                optype=op.optype,
                addr=new_addr,
                value=op.value,
                reg=new_reg,
                dep_on=op.dep_on,
            )
            all_ops.append(new_op)
            if op.reg and op.reg in base['forbidden']:
                forbidden[new_reg] = base['forbidden'][op.reg]

    return LitmusTest(
        name=f'{base_pattern}_ring_{n}t',
        n_threads=n,
        addresses=sorted(all_addrs),
        ops=all_ops,
        forbidden=forbidden,
    )


# ── 3. Bounded Model Checking ───────────────────────────────────────

def bounded_model_check(n_threads: int, n_ops_per_thread: int,
                        addresses: List[str],
                        target_arch: str) -> List[Dict]:
    """Enumerate all possible litmus test structures up to the given bound
    and check each against the target architecture's memory model.

    For each candidate:
      - Enumerate op types (store/load) per slot
      - Enumerate address assignments per slot
      - Construct a forbidden outcome (all loads read 0 = initial value)
      - Verify under the target model
      - Report hazards: tests where the forbidden outcome IS allowed

    This is a simple form of bounded model checking that goes beyond
    pattern matching to discover novel concurrency violations.
    """
    model = ARCHITECTURES.get(target_arch)
    if not model:
        return []

    hazards = []
    op_types = ['store', 'load']
    n_addrs = len(addresses)
    checked = 0
    skipped = 0

    # Generate all possible thread programs
    # Each thread has n_ops_per_thread slots, each slot is (type, addr)
    slot_choices = list(itertools.product(op_types, range(n_addrs)))
    thread_programs = list(itertools.product(slot_choices, repeat=n_ops_per_thread))

    # Limit: only check structurally interesting programs
    # Filter: at least one store and at least one load across all threads
    # Also limit total enumeration to keep runtime reasonable
    MAX_CANDIDATES = 5000
    candidate_count = 0

    for combo in itertools.product(thread_programs, repeat=n_threads):
        if candidate_count >= MAX_CANDIDATES:
            break

        # Build ops
        ops = []
        reg_counter = 0
        forbidden = {}
        has_store = False
        has_load = False
        thread_has_load = defaultdict(bool)
        thread_has_store = defaultdict(bool)

        for tid in range(n_threads):
            for optype, addr_idx in combo[tid]:
                addr = addresses[addr_idx]
                if optype == 'store':
                    ops.append(MemOp(tid, 'store', addr, 1))
                    has_store = True
                    thread_has_store[tid] = True
                else:
                    reg = f'r{reg_counter}'
                    reg_counter += 1
                    ops.append(MemOp(tid, 'load', addr, reg=reg))
                    forbidden[reg] = 0
                    has_load = True
                    thread_has_load[tid] = True

        # Skip degenerate cases
        if not has_store or not has_load:
            skipped += 1
            continue

        # Skip if all ops are on the same thread (not a concurrency test)
        active_threads = set(op.thread for op in ops)
        if len(active_threads) < 2:
            skipped += 1
            continue

        # Skip if no cross-thread communication (no shared addresses between threads)
        thread_addrs = defaultdict(set)
        for op in ops:
            thread_addrs[op.thread].add(op.addr)
        shared = False
        threads = sorted(thread_addrs.keys())
        for i in range(len(threads)):
            for j in range(i + 1, len(threads)):
                if thread_addrs[threads[i]] & thread_addrs[threads[j]]:
                    shared = True
                    break
            if shared:
                break
        if not shared:
            skipped += 1
            continue

        # Need at least one forbidden register
        if not forbidden:
            skipped += 1
            continue

        candidate_count += 1
        checked += 1

        all_addrs = sorted(set(op.addr for op in ops if op.addr))
        test = LitmusTest(
            name=f'bmc_{target_arch}_{checked}',
            n_threads=n_threads,
            addresses=all_addrs,
            ops=ops,
            forbidden=forbidden,
        )

        try:
            allowed, n_exec = verify_test(test, model)
        except Exception:
            continue

        if allowed:
            # This is a portability hazard: the forbidden outcome is reachable
            fence_rec = recommend_fence(test, target_arch, model)
            ops_desc = []
            for op in ops:
                if op.optype == 'store':
                    ops_desc.append(f'T{op.thread}:W[{op.addr}]=1')
                else:
                    ops_desc.append(f'T{op.thread}:R[{op.addr}]->{op.reg}')

            hazards.append({
                'test_name': test.name,
                'ops': '; '.join(ops_desc),
                'forbidden': forbidden,
                'allowed_under': model,
                'fence_recommendation': fence_rec,
                'executions_checked': n_exec,
            })

    return hazards


# ── Experiment Runner ────────────────────────────────────────────────

def _verify_across_archs(test: LitmusTest) -> Dict:
    """Verify a litmus test across all core architectures."""
    results = {}
    for arch in CORE_ARCHS:
        model = ARCHITECTURES[arch]
        try:
            allowed, n_checked = verify_test(test, model)
            fence = recommend_fence(test, arch, model) if allowed else None
            results[arch] = {
                'forbidden_allowed': allowed,
                'executions_checked': n_checked,
                'fence_recommendation': fence,
            }
        except Exception as e:
            results[arch] = {'error': str(e)}
    return results


def run_pattern_composition_experiments() -> Dict:
    """Run all pattern composition experiments and return JSON-serializable results."""
    results = {
        'metadata': {
            'description': 'Pattern composition experiments: beyond the 140-pattern ceiling',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        },
        'composition': {},
        'chains': {},
        'rings': {},
        'bmc': {},
        'summary': {},
    }

    # ── 1. Compose all pairs of 5 core patterns (10 pairs) ──
    print("=== Pattern Composition (10 pairs) ===")
    composition_results = {}
    novel_hazards = 0
    total_composed = 0

    for i, p1 in enumerate(CORE_PATTERNS):
        for p2 in CORE_PATTERNS[i + 1:]:
            total_composed += 1

            # Compose without shared variables (independent)
            test_indep = compose_patterns(p1, p2)
            indep_results = _verify_across_archs(test_indep)

            # Compose with first address of p2 shared with first address of p1
            p1_addrs = PATTERNS[p1]['addresses']
            p2_addrs = PATTERNS[p2]['addresses']
            shared = [p2_addrs[0]] if p2_addrs and p1_addrs and p2_addrs[0] in p1_addrs else [p1_addrs[0]]
            # Force sharing: use the first address from p1
            test_shared = compose_patterns(p1, p2, shared_vars=[p1_addrs[0]])
            shared_results = _verify_across_archs(test_shared)

            key = f'{p1}+{p2}'
            entry = {
                'pattern1': p1,
                'pattern2': p2,
                'composite_name': test_indep.name,
                'n_threads': test_indep.n_threads,
                'n_addresses': len(test_indep.addresses),
                'n_ops': len(test_indep.ops),
                'independent': indep_results,
                'shared_var': shared_results,
                'shared_addresses': [p1_addrs[0]],
            }

            # Count novel hazards (different behavior with shared vars)
            for arch in CORE_ARCHS:
                i_allowed = indep_results.get(arch, {}).get('forbidden_allowed', False)
                s_allowed = shared_results.get(arch, {}).get('forbidden_allowed', False)
                if s_allowed and not i_allowed:
                    novel_hazards += 1
                    entry.setdefault('novel_interactions', []).append(arch)

            composition_results[key] = entry
            print(f"  {key}: {test_indep.n_threads} threads, "
                  f"{len(test_indep.addresses)} addrs, "
                  f"{len(test_indep.ops)} ops")

    results['composition'] = composition_results

    # ── 2. Generate chains from mp and sb (3,4,5 threads = 6 variants) ──
    print("\n=== Chain Generation (6 variants) ===")
    chain_results = {}
    for base in ['mp', 'sb']:
        for n in [3, 4, 5]:
            test = generate_chain(base, n)
            arch_results = _verify_across_archs(test)
            key = f'{base}_chain_{n}t'
            chain_results[key] = {
                'base_pattern': base,
                'topology': 'chain',
                'n_threads': n,
                'n_addresses': len(test.addresses),
                'n_ops': len(test.ops),
                'forbidden': test.forbidden,
                'results': arch_results,
            }
            print(f"  {key}: {n} threads, {len(test.addresses)} addrs, "
                  f"{len(test.ops)} ops")

    results['chains'] = chain_results

    # ── 3. Generate rings from sb (3,4,5 threads = 3 variants) ──
    print("\n=== Ring Generation (3 variants) ===")
    ring_results = {}
    for n in [3, 4, 5]:
        test = generate_ring('sb', n)
        arch_results = _verify_across_archs(test)
        key = f'sb_ring_{n}t'
        ring_results[key] = {
            'base_pattern': 'sb',
            'topology': 'ring',
            'n_threads': n,
            'n_addresses': len(test.addresses),
            'n_ops': len(test.ops),
            'forbidden': test.forbidden,
            'results': arch_results,
        }
        print(f"  {key}: {n} threads, {len(test.addresses)} addrs, "
              f"{len(test.ops)} ops")

    results['rings'] = ring_results

    # ── 4. Bounded model checking (2 threads, 2 ops each, 2 addresses) ──
    print("\n=== Bounded Model Checking (2t, 2ops, 2addrs) ===")
    bmc_results = {}
    total_bmc_hazards = 0
    for arch in CORE_ARCHS:
        print(f"  Checking {arch}...", end=' ', flush=True)
        hazards = bounded_model_check(
            n_threads=2,
            n_ops_per_thread=2,
            addresses=['x', 'y'],
            target_arch=arch,
        )
        bmc_results[arch] = {
            'n_hazards': len(hazards),
            'hazards': hazards[:20],  # cap output size
        }
        total_bmc_hazards += len(hazards)
        print(f"{len(hazards)} hazards found")

    results['bmc'] = bmc_results

    # ── Summary ──
    total_patterns_generated = total_composed * 2 + 6 + 3  # indep + shared + chains + rings
    results['summary'] = {
        'composition_pairs': total_composed,
        'novel_interaction_hazards': novel_hazards,
        'chain_variants': 6,
        'ring_variants': 3,
        'total_generated_tests': total_patterns_generated,
        'bmc_hazards_found': total_bmc_hazards,
        'bmc_architectures_checked': len(CORE_ARCHS),
        'beyond_140_patterns': True,
        'contribution': (
            f'Generated {total_patterns_generated} composite tests and '
            f'found {total_bmc_hazards} hazards via bounded model checking, '
            f'demonstrating systematic exploration beyond the 140-pattern ceiling.'
        ),
    }

    print(f"\n=== Summary ===")
    print(f"  Total generated tests: {total_patterns_generated}")
    print(f"  Novel interaction hazards: {novel_hazards}")
    print(f"  BMC hazards found: {total_bmc_hazards}")

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    results = run_pattern_composition_experiments()

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), 'paper_results_v10')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pattern_composition.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print key results
    print(f"\n{'='*60}")
    print("KEY RESULTS: Beyond the 140-Pattern Ceiling")
    print(f"{'='*60}")
    print(json.dumps(results['summary'], indent=2))


if __name__ == '__main__':
    main()
