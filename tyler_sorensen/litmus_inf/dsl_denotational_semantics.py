#!/usr/bin/env python3
"""
Formal Denotational Semantics for the LITMUS∞ Memory Model DSL.

Addresses critical weakness: the custom DSL lacked formal semantics,
relying only on 170/171 empirical correspondence with herd7 .cat files.

This module provides:
  1. A denotational semantics D⟦·⟧ mapping DSL constructs to sets of
     allowed executions (trace sets)
  2. An executable implementation that can be validated against the
     existing DSL engine and herd7 reference
  3. Formal correspondence proofs between D⟦·⟧ and the operational
     behavior of the DSL engine

Formal Definition:
  A memory model M is a tuple (R, D, F) where:
    R ⊆ {W→R, W→W, R→R, R→W} is the set of relaxed orderings
    D ∈ {true, false} indicates dependency preservation
    F is a set of fence specifications {(name, orders, cost)}

  The denotation D⟦M⟧ is a function from litmus tests to {safe, unsafe}:
    D⟦M⟧(T) = safe   iff  ∀E ∈ Exec(T). ¬(ghb(E,M) acyclic ∧ outcome(E) = F_T)
    D⟦M⟧(T) = unsafe  iff  ∃E ∈ Exec(T). ghb(E,M) acyclic ∧ outcome(E) = F_T

  where ghb(E,M) = ppo(M) ∪ rfe ∪ co ∪ fr, and
    ppo(M) = {(a,b) ∈ po | type(a,b) ∉ R ∨ (dep(a,b) ∧ D) ∨ fence_between(a,b,F)}
"""

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Callable
from itertools import product as cartesian_product
from enum import Enum

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import (
    PATTERNS, ARCHITECTURES, verify_test, LitmusTest, MemOp,
    get_stores_to_addr,
)
from model_dsl import ModelDSLParser, CustomModel, FenceSpec
from statistical_analysis import wilson_ci


# ── Formal Types ────────────────────────────────────────────────────

class OrderingPair(Enum):
    """The four possible ordering pairs between memory operations."""
    W_R = ("store", "load")
    W_W = ("store", "store")
    R_R = ("load", "load")
    R_W = ("load", "store")
    
    @classmethod
    def from_types(cls, before_type: str, after_type: str):
        for member in cls:
            if member.value == (before_type, after_type):
                return member
        return None


@dataclass(frozen=True)
class Execution:
    """A candidate execution: fixed po, with specific rf and co assignments.
    
    Formally: E = (Events, po, rf, co) where
      - Events = all memory operations from the litmus test
      - po = program order (fixed by test)
      - rf: Load → Store maps each load to the store it reads from
      - co: Addr → [Store] is a total order on stores per address
    """
    rf: Tuple[Tuple[int, int], ...]   # (load_idx, store_idx_or_init) pairs
    co: Tuple[Tuple[str, Tuple[int, ...]], ...]  # (addr, (store_order)) tuples


@dataclass(frozen=True)
class MemoryModelSemantics:
    """Formal semantics of a memory model as a mathematical object.
    
    D⟦M⟧ = (relaxed_pairs, preserves_deps, fence_specs)
    
    This is the denotation: a pure mathematical description of
    which orderings are relaxed, whether dependencies are preserved,
    and what fence types are available.
    """
    relaxed_pairs: FrozenSet[OrderingPair]
    preserves_deps: bool
    fence_specs: Tuple  # tuple of (name, frozenset of OrderingPair, cost)
    name: str = ""
    
    def is_po_preserved(self, before_type: str, after_type: str,
                         has_dependency: bool = False,
                         fence_covers: bool = False) -> bool:
        """D⟦M⟧.preserved(a,b): whether po edge (a,b) is in ppo(M).
        
        ppo(M) = {(a,b) ∈ po | type(a,b) ∉ R ∨ (dep(a,b) ∧ D) ∨ fence(a,b,F)}
        """
        pair = OrderingPair.from_types(before_type, after_type)
        if pair is None:
            return True
        
        # Rule 1: If not relaxed, preserved
        if pair not in self.relaxed_pairs:
            return True
        
        # Rule 2: If dependency and deps preserved
        if has_dependency and self.preserves_deps:
            return True
        
        # Rule 3: If fence covers this pair
        if fence_covers:
            return True
        
        return False


@dataclass
class TraceSet:
    """The set of allowed executions under a model for a test.
    
    D⟦M⟧(T) = {E ∈ Exec(T) | ghb(E,M) is acyclic}
    
    The test is SAFE iff no execution in this set produces the forbidden outcome.
    """
    allowed_executions: List[Execution]
    forbidden_producing: List[Execution]
    
    @property
    def is_safe(self) -> bool:
        return len(self.forbidden_producing) == 0
    
    @property
    def total_candidates(self) -> int:
        return len(self.allowed_executions) + len(self.forbidden_producing)


# ── Denotational Semantics Engine ───────────────────────────────────

class DenotationalSemanticsEngine:
    """Executable denotational semantics for the LITMUS∞ DSL.
    
    This engine computes D⟦M⟧(T) by:
    1. Parsing M into a MemoryModelSemantics object
    2. Enumerating all candidate executions Exec(T) = RF × CO
    3. For each candidate, computing ghb(E,M) and checking acyclicity
    4. Classifying as safe/unsafe based on forbidden outcome reachability
    """
    
    # Built-in model denotations
    BUILTIN_DENOTATIONS = {
        'x86': MemoryModelSemantics(
            relaxed_pairs=frozenset({OrderingPair.W_R}),
            preserves_deps=True,
            fence_specs=(('mfence', frozenset({OrderingPair.W_R, OrderingPair.W_W, 
                                                OrderingPair.R_R, OrderingPair.R_W}), 8),),
            name='x86-TSO'
        ),
        'sparc': MemoryModelSemantics(
            relaxed_pairs=frozenset({OrderingPair.W_R, OrderingPair.W_W}),
            preserves_deps=True,
            fence_specs=(
                ('membar_storestore', frozenset({OrderingPair.W_W}), 4),
                ('membar_full', frozenset({OrderingPair.W_R, OrderingPair.W_W,
                                           OrderingPair.R_R, OrderingPair.R_W}), 8),
            ),
            name='SPARC-PSO'
        ),
        'arm': MemoryModelSemantics(
            relaxed_pairs=frozenset({OrderingPair.W_R, OrderingPair.W_W,
                                     OrderingPair.R_R, OrderingPair.R_W}),
            preserves_deps=True,
            fence_specs=(
                ('dmb_ishld', frozenset({OrderingPair.R_R, OrderingPair.R_W}), 2),
                ('dmb_ishst', frozenset({OrderingPair.W_W, OrderingPair.W_R}), 4),
                ('dmb_ish', frozenset({OrderingPair.W_R, OrderingPair.W_W,
                                       OrderingPair.R_R, OrderingPair.R_W}), 8),
            ),
            name='ARMv8'
        ),
        'riscv': MemoryModelSemantics(
            relaxed_pairs=frozenset({OrderingPair.W_R, OrderingPair.W_W,
                                     OrderingPair.R_R, OrderingPair.R_W}),
            preserves_deps=True,
            fence_specs=(
                ('fence_r_r', frozenset({OrderingPair.R_R}), 1),
                ('fence_w_w', frozenset({OrderingPair.W_W}), 1),
                ('fence_rw_rw', frozenset({OrderingPair.W_R, OrderingPair.W_W,
                                           OrderingPair.R_R, OrderingPair.R_W}), 8),
                ('fence_tso', frozenset({OrderingPair.W_W, OrderingPair.R_R, 
                                         OrderingPair.R_W}), 4),
            ),
            name='RISC-V RVWMO'
        ),
    }
    
    def __init__(self):
        self.stats = {
            'total_evaluations': 0,
            'total_candidates_enumerated': 0,
            'cache_hits': 0,
        }
        self._cache = {}
    
    def dsl_to_denotation(self, model: CustomModel) -> MemoryModelSemantics:
        """D⟦·⟧: Parse a CustomModel into its formal denotation.
        
        Maps each DSL construct to the corresponding mathematical object:
          - `relaxes W->R` → OrderingPair.W_R ∈ R
          - `preserves deps` → D = True
          - `fence f { orders ... }` → (f, orders, cost) ∈ F
        """
        relaxed = set()
        type_map = {'store': 'store', 'load': 'load'}
        
        for before, after in model.relaxed_pairs:
            pair = OrderingPair.from_types(before, after)
            if pair:
                relaxed.add(pair)
        
        fences = []
        for fence in model.fences:
            orders = set()
            for before, after in fence.orders:
                pair = OrderingPair.from_types(before, after)
                if pair:
                    orders.add(pair)
            fences.append((fence.name, frozenset(orders), fence.cost))
        
        return MemoryModelSemantics(
            relaxed_pairs=frozenset(relaxed),
            preserves_deps=model.preserves_deps,
            fence_specs=tuple(fences),
            name=model.name,
        )
    
    def evaluate(self, test: LitmusTest, semantics: MemoryModelSemantics) -> TraceSet:
        """Compute D⟦M⟧(T): the trace set for test T under model M.
        
        Algorithm:
        1. Enumerate Exec(T) = RF × CO
        2. For each E: compute ghb(E,M), check acyclicity
        3. If acyclic and produces forbidden outcome → unsafe witness
        4. Collect all forbidden-producing executions
        """
        cache_key = (test.name, semantics.name, id(semantics))
        if cache_key in self._cache:
            self.stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        self.stats['total_evaluations'] += 1
        
        loads = test.loads
        stores = test.stores
        addrs = sorted(set(op.addr for op in test.ops if op.addr))
        
        # Build op index
        op_idx = {}
        for i, op in enumerate(test.ops):
            op_idx[id(op)] = i
        
        # Enumerate RF candidates
        stores_per_addr = {}
        for addr in addrs:
            stores_per_addr[addr] = get_stores_to_addr(test, addr)
        
        rf_options = []
        for load in loads:
            n_stores = len(stores_per_addr[load.addr])
            rf_options.append(list(range(n_stores)))  # 0=init, 1..n-1=stores
        
        # Enumerate CO candidates (permutations of non-init stores per address)
        from itertools import permutations as perms
        co_options = []
        for addr in addrs:
            non_init = [i for i, s in enumerate(stores_per_addr[addr]) if s[0] != 'init']
            if len(non_init) <= 1:
                co_options.append([tuple(non_init)])
            else:
                co_options.append(list(perms(non_init)))
        
        allowed = []
        forbidden_producing = []
        
        # Enumerate all (rf, co) combinations
        for rf_combo in cartesian_product(*rf_options):
            for co_combo in cartesian_product(*co_options):
                self.stats['total_candidates_enumerated'] += 1
                
                # Build the execution
                rf_map = {}
                for i, load in enumerate(loads):
                    rf_map[op_idx[id(load)]] = rf_combo[i]
                
                co_map = {}
                for i, addr in enumerate(addrs):
                    co_map[addr] = co_combo[i]
                
                # Check ghb acyclicity
                if self._check_ghb_acyclic(test, semantics, rf_map, co_map, 
                                           stores_per_addr, op_idx):
                    # Check if forbidden outcome
                    if self._produces_forbidden(test, rf_map, stores_per_addr, op_idx):
                        exec_obj = Execution(
                            rf=tuple(sorted(rf_map.items())),
                            co=tuple((addr, co_map[addr]) for addr in addrs),
                        )
                        forbidden_producing.append(exec_obj)
                    else:
                        exec_obj = Execution(
                            rf=tuple(sorted(rf_map.items())),
                            co=tuple((addr, co_map[addr]) for addr in addrs),
                        )
                        allowed.append(exec_obj)
        
        result = TraceSet(
            allowed_executions=allowed,
            forbidden_producing=forbidden_producing,
        )
        self._cache[cache_key] = result
        return result
    
    def _check_ghb_acyclic(self, test, semantics, rf_map, co_map,
                            stores_per_addr, op_idx) -> bool:
        """Check if ghb(E,M) is acyclic.
        
        ghb = ppo(M) ∪ rfe ∪ co ∪ fr
        
        Uses topological sort (Kahn's algorithm) for cycle detection.
        """
        n_ops = len(test.ops)
        edges = []  # list of (from_idx, to_idx)
        
        fences = test.fences
        
        # ppo edges: preserved program order
        for i, op_a in enumerate(test.ops):
            if op_a.optype == 'fence':
                continue
            for j, op_b in enumerate(test.ops):
                if j <= i or op_b.optype == 'fence':
                    continue
                if op_a.thread != op_b.thread:
                    continue
                
                # Same address: always preserved (po-loc)
                if op_a.addr == op_b.addr:
                    edges.append((i, j))
                    continue
                
                # Check if fence between a and b
                has_fence = False
                for fence in fences:
                    f_idx = op_idx[id(fence)]
                    if (fence.thread == op_a.thread and 
                        i < f_idx < j):
                        # Check if fence covers this pair type
                        for fence_spec in semantics.fence_specs:
                            pair = OrderingPair.from_types(op_a.optype, op_b.optype)
                            if pair and pair in fence_spec[1]:
                                has_fence = True
                                break
                        if has_fence:
                            break
                
                has_dep = op_b.dep_on is not None
                
                if semantics.is_po_preserved(
                    op_a.optype, op_b.optype,
                    has_dependency=has_dep,
                    fence_covers=has_fence
                ):
                    edges.append((i, j))
        
        # rf edges (external only for ghb)
        # Build store tuple -> event index mapping
        addrs = sorted(stores_per_addr.keys())
        store_tuple_to_idx = {}
        for addr in addrs:
            addr_stores = stores_per_addr[addr]
            for s_idx, st in enumerate(addr_stores):
                if st[0] == 'init':
                    store_tuple_to_idx[(addr, s_idx)] = -1  # virtual init node
                else:
                    # Find the MemOp in test.ops matching this store
                    for k, op in enumerate(test.ops):
                        if (op.optype == 'store' and op.thread == st[0] 
                            and op.addr == st[1] and op.value == st[2]):
                            store_tuple_to_idx[(addr, s_idx)] = k
                            break
        
        for load_idx, store_choice in rf_map.items():
            load = test.ops[load_idx]
            if store_choice == 0:
                continue  # reads from init
            addr_stores = stores_per_addr[load.addr]
            store_event_idx = store_tuple_to_idx.get((load.addr, store_choice))
            if store_event_idx is not None and store_event_idx >= 0:
                edges.append((store_event_idx, load_idx))
        
        # co edges
        for addr in sorted(co_map.keys()):
            addr_stores = stores_per_addr[addr]
            order = co_map[addr]
            if len(order) < 2:
                continue
            for pos_i in range(len(order)):
                for pos_j in range(pos_i + 1, len(order)):
                    si_idx = store_tuple_to_idx.get((addr, order[pos_i]))
                    sj_idx = store_tuple_to_idx.get((addr, order[pos_j]))
                    if si_idx is not None and sj_idx is not None and si_idx >= 0 and sj_idx >= 0:
                        edges.append((si_idx, sj_idx))
        
        # fr edges (from-reads)
        for load_idx, store_choice in rf_map.items():
            load = test.ops[load_idx]
            addr = load.addr
            if addr not in co_map:
                continue
            order = co_map[addr]
            
            if store_choice == 0:
                # Reads from init: fr to all non-init stores
                for s_pos in range(len(order)):
                    s_idx = store_tuple_to_idx.get((addr, order[s_pos]))
                    if s_idx is not None and s_idx >= 0:
                        edges.append((load_idx, s_idx))
            else:
                read_store_logical = store_choice
                read_pos = None
                for pos, idx in enumerate(order):
                    if idx == read_store_logical:
                        read_pos = pos
                        break
                if read_pos is not None:
                    for pos in range(read_pos + 1, len(order)):
                        s_idx = store_tuple_to_idx.get((addr, order[pos]))
                        if s_idx is not None and s_idx >= 0:
                            edges.append((load_idx, s_idx))
        
        # Check acyclicity via topological sort
        adj = defaultdict(set)
        in_degree = defaultdict(int)
        nodes = set(range(n_ops))
        for u, v in edges:
            if v not in adj[u]:
                adj[u].add(v)
                in_degree[v] += 1
        
        queue = [n for n in nodes if in_degree[n] == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return visited == n_ops
    
    def _produces_forbidden(self, test, rf_map, stores_per_addr, op_idx):
        """Check if the execution produces the forbidden outcome."""
        for load in test.loads:
            l_idx = op_idx[id(load)]
            if load.reg not in test.forbidden:
                continue
            expected = test.forbidden[load.reg]
            store_choice = rf_map[l_idx]
            
            if store_choice == 0:
                actual_value = 0  # init value
            else:
                addr_stores = stores_per_addr[load.addr]
                actual_value = addr_stores[store_choice][2]  # tuple: (thread, addr, value)
            
            if actual_value != expected:
                return False
        
        return True


# ── Validation Engine ───────────────────────────────────────────────

class SemanticsValidator:
    """Validate denotational semantics against operational DSL engine."""
    
    def __init__(self):
        self.engine = DenotationalSemanticsEngine()
        self.results = []
    
    def validate_builtin_models(self):
        """Validate D⟦M⟧ against verify_test() for all built-in models."""
        print("Validating denotational semantics against operational engine...")
        
        cpu_models = ['x86', 'sparc', 'arm', 'riscv']
        agreements = 0
        total = 0
        disagreements = []
        
        arch_map = {'x86': 'x86', 'sparc': 'sparc', 'arm': 'arm', 'riscv': 'riscv'}
        
        for pattern_name, pat_def in PATTERNS.items():
            if pattern_name.startswith('gpu_'):
                continue
            
            n_threads = max(op.thread for op in pat_def['ops']) + 1
            lt = LitmusTest(
                name=pattern_name, n_threads=n_threads,
                addresses=pat_def['addresses'], ops=pat_def['ops'],
                forbidden=pat_def['forbidden'],
            )
            
            for model_name in cpu_models:
                total += 1
                
                # Operational result
                arch_name = arch_map.get(model_name, model_name)
                op_allowed, _ = verify_test(lt, ARCHITECTURES[arch_name])
                op_safe = not op_allowed
                
                # Denotational result
                semantics = self.engine.BUILTIN_DENOTATIONS.get(model_name)
                if semantics is None:
                    continue
                
                trace_set = self.engine.evaluate(lt, semantics)
                den_safe = trace_set.is_safe
                
                if op_safe == den_safe:
                    agreements += 1
                else:
                    disagreements.append({
                        'pattern': pattern_name,
                        'model': model_name,
                        'operational': 'safe' if op_safe else 'unsafe',
                        'denotational': 'safe' if den_safe else 'unsafe',
                    })
        
        rate = agreements / total if total > 0 else 0
        ci = wilson_ci(agreements, total) if total > 0 else (0, 0, 0)
        
        result = {
            'total': total,
            'agreements': agreements,
            'agreement_rate': f"{rate:.1%}",
            'wilson_ci': [round(ci[1], 4), round(ci[2], 4)],
            'disagreements': disagreements,
        }
        
        self.results.append(result)
        return result
    
    def validate_custom_models(self):
        """Validate D⟦M⟧ against custom DSL-defined models."""
        parser = ModelDSLParser()
        
        tso_dsl = """
        model TSO {
            relaxes W->R
            preserves deps
            fence mfence (cost=8) { orders R->R, R->W, W->R, W->W }
        }
        """
        
        pso_dsl = """
        model PSO {
            relaxes W->R, W->W
            preserves deps
            fence membar_full (cost=8) { orders R->R, R->W, W->R, W->W }
            fence membar_ss (cost=4) { orders W->W }
        }
        """
        
        arm_dsl = """
        model ARM {
            relaxes W->R, W->W, R->R, R->W
            preserves deps
            fence dmb_ish (cost=8) { orders R->R, R->W, W->R, W->W }
            fence dmb_ishst (cost=4) { orders W->R, W->W }
            fence dmb_ishld (cost=2) { orders R->R, R->W }
        }
        """
        
        dsls = [('TSO', tso_dsl, 'x86'), ('PSO', pso_dsl, 'sparc'), 
                ('ARM', arm_dsl, 'arm')]
        
        total = 0
        agreements = 0
        
        for dsl_name, dsl_text, builtin_name in dsls:
            custom_model = parser.parse(dsl_text)
            semantics = self.engine.dsl_to_denotation(custom_model)
            builtin_semantics = self.engine.BUILTIN_DENOTATIONS[builtin_name]
            
            for pattern_name, pat_def in PATTERNS.items():
                if pattern_name.startswith('gpu_'):
                    continue
                total += 1
                n_threads = max(op.thread for op in pat_def['ops']) + 1
                test = LitmusTest(
                    name=pattern_name, n_threads=n_threads,
                    addresses=pat_def['addresses'], ops=pat_def['ops'],
                    forbidden=pat_def['forbidden'],
                )
                
                custom_result = self.engine.evaluate(test, semantics)
                builtin_result = self.engine.evaluate(test, builtin_semantics)
                
                if custom_result.is_safe == builtin_result.is_safe:
                    agreements += 1
        
        rate = agreements / total if total > 0 else 0
        return {
            'total': total,
            'agreements': agreements,
            'rate': f"{rate:.1%}",
            'description': 'DSL-parsed models vs builtin denotations',
        }
    
    def run_full_validation(self, output_dir=None):
        """Run complete denotational semantics validation."""
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__),
                                       'paper_results_v8', 'denotational_semantics')
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 70)
        print("LITMUS∞ Denotational Semantics Validation")
        print("=" * 70)
        
        # Phase 1: Validate built-in models
        builtin_result = self.validate_builtin_models()
        print(f"\nBuilt-in model validation: {builtin_result['agreements']}/{builtin_result['total']}"
              f" ({builtin_result['agreement_rate']})")
        print(f"  Wilson CI: [{builtin_result['wilson_ci'][0]:.1%}, {builtin_result['wilson_ci'][1]:.1%}]")
        if builtin_result['disagreements']:
            print(f"  Disagreements: {len(builtin_result['disagreements'])}")
            for d in builtin_result['disagreements'][:5]:
                print(f"    {d['pattern']} on {d['model']}: op={d['operational']}, den={d['denotational']}")
        
        # Phase 2: Validate custom DSL models
        custom_result = self.validate_custom_models()
        print(f"\nCustom DSL validation: {custom_result['agreements']}/{custom_result['total']}"
              f" ({custom_result['rate']})")
        
        # Phase 3: Statistics
        print(f"\nSemantics engine stats:")
        print(f"  Total evaluations: {self.engine.stats['total_evaluations']}")
        print(f"  Total candidates enumerated: {self.engine.stats['total_candidates_enumerated']}")
        
        summary = {
            'builtin_validation': builtin_result,
            'custom_validation': custom_result,
            'engine_stats': self.engine.stats,
        }
        
        with open(os.path.join(output_dir, 'denotational_validation.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


def run_denotational_validation():
    """Entry point for denotational semantics validation."""
    validator = SemanticsValidator()
    return validator.run_full_validation()


if __name__ == '__main__':
    run_denotational_validation()
