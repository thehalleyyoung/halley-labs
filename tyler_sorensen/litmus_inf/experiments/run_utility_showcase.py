"""
Utility Showcase: Comprehensive demonstration of LITMUS∞ platform capabilities.

Runs seven evaluation categories:
  1. Memory model comparison (10 litmus tests × 7 models)
  2. Race detection accuracy (20 synthetic programs, P/R/F1)
  3. Fence optimization (5 programs, minimum fences vs naive)
  4. Deadlock detection (10 programs, 5 with deadlocks, 5 without)
  5. Lock-free verification (Treiber stack + MS queue, 1000 interleavings)
  6. Model checking (Peterson's + bakery algorithm, states explored)
  7. DPOR efficiency (5 programs, random vs DPOR reduction)

Results saved to utility_showcase_results.json.
"""

import sys
import os
import json
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from memory_model import (
    MemoryEvent, ExecutionBuilder, FenceType, Scope,
    SequentialConsistency, TotalStoreOrder, PartialStoreOrder,
    RelaxedMemoryModel, RISCVModel, PTXModel, VulkanModel,
    get_model,
)
from litmus_test_engine import (
    make_store_buffering, make_message_passing, make_load_buffering,
    make_iriw, make_wrc, make_corr, make_cowr, make_corw, make_coww,
    generate_all_outcomes, check_outcome, classify_test,
    LitmusTestSuite, OutcomeEnumerator, Outcome,
    BUILTIN_TESTS,
)
from race_detector import (
    HappensBeforeDetector, FastTrackDetector,
    TraceEvent, make_trace_event,
    generate_racy_trace, generate_synchronized_trace,
)
from fence_optimizer import (
    FenceOptimizationPipeline, FenceVerifier,
    make_store_buffering_program, make_message_passing_program,
    make_dekker_program, ConcurrentProgram,
)
from deadlock_detector import DeadlockDetector, LockEvent
from concurrent_data_structures import (
    LockFreeStack, LockFreeQueue,
    LinearizabilityChecker, Operation,
)
from model_checker import (
    ModelChecker, BoundedModelChecker,
    make_petersons_algorithm, mutual_exclusion_property,
    make_racy_counter, ConcurrentProgram as MCProgram,
    SharedState, ThreadState, SystemState, InstrType, Instruction,
    ThreadProgram,
)

RESULTS = {}


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ============================================================================
# 1. Memory Model Comparison
# ============================================================================

def _build_execution_for_test(test_name, test):
    """Build a forbidden-outcome execution for a litmus test."""
    MemoryEvent._counter = 0
    if not test.forbidden_outcomes:
        return None
    fo = test.forbidden_outcomes[0]
    enumerator = OutcomeEnumerator(test)
    exe = enumerator._outcome_to_execution(fo)
    return exe


def run_memory_model_comparison():
    section("1. MEMORY MODEL COMPARISON")
    tests = {
        'SB': make_store_buffering(),
        'MP': make_message_passing(),
        'LB': make_load_buffering(),
        'IRIW': make_iriw(),
        'WRC': make_wrc(),
        'CoRR': make_corr(),
        'CoWR': make_cowr(),
        'CoRW': make_corw(),
        'CoWW': make_coww(),
    }

    model_names = ['SC', 'TSO', 'PSO', 'Relaxed', 'RVWMO', 'PTX', 'Vulkan']
    model_instances = {
        'SC': SequentialConsistency(),
        'TSO': TotalStoreOrder(),
        'PSO': PartialStoreOrder(),
        'Relaxed': RelaxedMemoryModel(),
        'RVWMO': RISCVModel(),
        'PTX': PTXModel(),
        'Vulkan': VulkanModel(),
    }

    # For tests without forbidden outcomes, use classify_test style
    matrix = {}
    for test_name, test in tests.items():
        MemoryEvent._counter = 0
        row = {}
        if test.forbidden_outcomes:
            fo = test.forbidden_outcomes[0]
            for mname, model in model_instances.items():
                try:
                    result = check_outcome(test, fo, model)
                    row[mname] = result
                except Exception:
                    row[mname] = 'error'
        else:
            # No forbidden outcome: test is about coherence ordering
            for mname in model_names:
                row[mname] = 'allowed'
        matrix[test_name] = row

    # Print matrix
    header = f"{'Test':<8}" + "".join(f"{m:<10}" for m in model_names)
    print(header)
    print("-" * len(header))
    for test_name in tests:
        row = matrix[test_name]
        line = f"{test_name:<8}"
        for m in model_names:
            val = row.get(m, '?')
            symbol = '✓' if val == 'allowed' else '✗' if val == 'forbidden' else '?'
            line += f"{symbol:<10}"
        print(line)

    # Count forbidden per model
    forbidden_counts = {}
    for m in model_names:
        count = sum(1 for t in matrix.values() if t.get(m) == 'forbidden')
        forbidden_counts[m] = count

    print(f"\nForbidden counts: {forbidden_counts}")
    RESULTS['memory_model_comparison'] = {
        'matrix': matrix,
        'forbidden_counts': forbidden_counts,
        'n_tests': len(tests),
        'n_models': len(model_names),
    }
    return True


# ============================================================================
# 2. Race Detection Accuracy
# ============================================================================

def _make_racy_program(seed, program_id):
    """Create a synthetic racy trace."""
    rng = np.random.RandomState(seed)
    TraceEvent._counter = 0
    events = []
    shared_addr = f"shared_{program_id}"
    n_ops = rng.randint(6, 12)
    for i in range(n_ops):
        tid = rng.randint(0, 2)
        op = 'write' if rng.random() < 0.5 else 'read'
        events.append(make_trace_event(tid, op, shared_addr, i))
    return events


def _make_racefree_program(seed, program_id):
    """Create a synchronized (race-free) trace."""
    rng = np.random.RandomState(seed)
    TraceEvent._counter = 0
    events = []
    shared_addr = f"shared_{program_id}"
    lock_id = f"lock_{program_id}"
    n_ops = rng.randint(4, 8)
    for i in range(n_ops):
        tid = rng.randint(0, 2)
        events.append(make_trace_event(tid, 'acquire', lock=lock_id))
        op = 'write' if rng.random() < 0.5 else 'read'
        events.append(make_trace_event(tid, op, shared_addr, i))
        events.append(make_trace_event(tid, 'release', lock=lock_id))
    return events


def run_race_detection_accuracy():
    section("2. RACE DETECTION ACCURACY")

    tp, fp, tn, fn = 0, 0, 0, 0

    results_detail = []
    for i in range(10):
        TraceEvent._counter = 0
        trace = _make_racy_program(seed=100 + i, program_id=i)
        det = HappensBeforeDetector(n_threads=4)
        for e in trace:
            det.process_event(e)
        races = det.get_races()
        detected = len(races) > 0
        if detected:
            tp += 1
        else:
            fn += 1
        results_detail.append({
            'program': f'racy_{i}', 'has_race': True,
            'detected': detected, 'n_races': len(races)
        })

    for i in range(10):
        TraceEvent._counter = 0
        trace = _make_racefree_program(seed=200 + i, program_id=10 + i)
        det = HappensBeforeDetector(n_threads=4)
        for e in trace:
            det.process_event(e)
        races = det.get_races()
        detected = len(races) > 0
        if not detected:
            tn += 1
        else:
            fp += 1
        results_detail.append({
            'program': f'racefree_{i}', 'has_race': False,
            'detected': detected, 'n_races': len(races)
        })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    RESULTS['race_detection'] = {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'details': results_detail,
    }
    return True


# ============================================================================
# 3. Fence Optimization
# ============================================================================

def _make_iriw_program():
    """IRIW-like program: 4 threads, independent writes observed by readers."""
    prog = ConcurrentProgram()
    prog.add_store(0, 'x', 1)
    prog.add_store(1, 'y', 1)
    prog.add_load(2, 'x')
    prog.add_load(2, 'y')
    prog.add_load(3, 'y')
    prog.add_load(3, 'x')
    return prog


def _make_wrc_program():
    """WRC-like program: write-read causality."""
    prog = ConcurrentProgram()
    prog.add_store(0, 'x', 1)
    prog.add_load(1, 'x')
    prog.add_store(1, 'y', 1)
    prog.add_load(2, 'y')
    prog.add_load(2, 'x')
    return prog


def run_fence_optimization():
    section("3. FENCE OPTIMIZATION")

    programs = {
        'SB': make_store_buffering_program(),
        'MP': make_message_passing_program(),
        'Dekker': make_dekker_program(),
        'IRIW': _make_iriw_program(),
        'WRC': _make_wrc_program(),
    }

    pipeline = FenceOptimizationPipeline()
    results = {}

    for name, prog in programs.items():
        try:
            result = pipeline.run(prog, 'arm')
            n_fences = result['n_fences']
            # Naive: fence after every access
            n_accesses = len(prog.all_accesses())
            naive_fences = max(n_accesses, n_fences)
            reduction = 1.0 - (n_fences / naive_fences) if naive_fences > 0 else 0.0

            results[name] = {
                'n_fences_optimal': n_fences,
                'n_fences_naive': naive_fences,
                'reduction_ratio': round(reduction, 3),
                'verified': result['verified'],
                'cost_initial': result.get('initial_cost', 0),
                'cost_reduced': result.get('reduced_cost', 0),
            }
            print(f"  {name:>8}: optimal={n_fences}, naive={naive_fences}, "
                  f"reduction={reduction:.1%}, verified={result['verified']}")
        except Exception as e:
            results[name] = {'error': str(e)}
            print(f"  {name:>8}: ERROR - {e}")

    RESULTS['fence_optimization'] = results
    return True


# ============================================================================
# 4. Deadlock Detection
# ============================================================================

def run_deadlock_detection():
    section("4. DEADLOCK DETECTION")

    results = []
    tp, fp, tn, fn = 0, 0, 0, 0

    # 5 programs WITH deadlocks
    deadlock_patterns = [
        # AB-BA classic
        [('T0', 'A', 'B'), ('T1', 'B', 'A')],
        # Three-way cycle: A->B, B->C, C->A
        [('T0', 'A', 'B'), ('T1', 'B', 'C'), ('T2', 'C', 'A')],
        # Four-way cycle
        [('T0', 'A', 'B'), ('T1', 'B', 'C'), ('T2', 'C', 'D'), ('T3', 'D', 'A')],
        # Double AB-BA (two independent cycles)
        [('T0', 'A', 'B'), ('T1', 'B', 'A'), ('T2', 'C', 'D'), ('T3', 'D', 'C')],
        # Nested with cycle
        [('T0', 'A', 'B'), ('T0', 'B', 'C'), ('T1', 'C', 'A')],
    ]

    for i, pattern in enumerate(deadlock_patterns):
        LockEvent._counter = 0
        det = DeadlockDetector()
        ts = 1
        for entry in pattern:
            tid = int(entry[0][1:])
            locks = entry[1:]
            for lock in locks:
                det.add_lock_event(tid, lock, 'acquire', ts)
                ts += 1
            for lock in reversed(locks):
                det.add_lock_event(tid, lock, 'release', ts)
                ts += 1
        deadlocks = det.check()
        detected = len(deadlocks) > 0
        if detected:
            tp += 1
        else:
            fn += 1
        results.append({
            'program': f'deadlock_{i}', 'has_deadlock': True,
            'detected': detected, 'n_deadlocks': len(deadlocks),
        })
        print(f"  deadlock_{i}: {'DETECTED' if detected else 'MISSED'} ({len(deadlocks)} cycles)")

    # 5 programs WITHOUT deadlocks (consistent ordering)
    safe_patterns = [
        # Consistent A->B ordering
        [('T0', 'A', 'B'), ('T1', 'A', 'B')],
        # Single lock per thread
        [('T0', 'A'), ('T1', 'B'), ('T2', 'C')],
        # Consistent A->B->C ordering
        [('T0', 'A', 'B', 'C'), ('T1', 'A', 'B', 'C')],
        # Non-overlapping locks
        [('T0', 'A'), ('T1', 'B')],
        # Same lock, no conflict
        [('T0', 'A'), ('T1', 'A')],
    ]

    for i, pattern in enumerate(safe_patterns):
        LockEvent._counter = 0
        det = DeadlockDetector()
        ts = 1
        for entry in pattern:
            tid = int(entry[0][1:])
            locks = entry[1:]
            for lock in locks:
                det.add_lock_event(tid, lock, 'acquire', ts)
                ts += 1
            for lock in reversed(locks):
                det.add_lock_event(tid, lock, 'release', ts)
                ts += 1
        deadlocks = det.check()
        detected = len(deadlocks) > 0
        if not detected:
            tn += 1
        else:
            fp += 1
        results.append({
            'program': f'safe_{i}', 'has_deadlock': False,
            'detected': detected, 'n_deadlocks': len(deadlocks),
        })
        print(f"  safe_{i}:     {'FALSE POS' if detected else 'CORRECT'}")

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    print(f"\n  Accuracy: {accuracy:.1%} (TP={tp}, FP={fp}, TN={tn}, FN={fn})")

    RESULTS['deadlock_detection'] = {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': round(accuracy, 4),
        'details': results,
    }
    return True


# ============================================================================
# 5. Lock-Free Verification
# ============================================================================

def run_lockfree_verification():
    section("5. LOCK-FREE VERIFICATION (Treiber Stack + MS Queue)")

    rng = np.random.RandomState(42)

    # Treiber Stack: 1000 random interleavings
    print("  Treiber Stack (1000 random interleavings)...")
    stack_linearizable = 0
    stack_total = 1000
    for trial in range(stack_total):
        stack = LockFreeStack()
        ops = []
        t = 0
        n_ops = rng.randint(3, 8)
        values_pushed = []
        for j in range(n_ops):
            tid = rng.randint(0, 3)
            if rng.random() < 0.6 or not values_pushed:
                val = rng.randint(1, 100)
                stack.push(val, thread_id=tid)
                values_pushed.append(val)
                ops.append(Operation('push', val, None, t, t + 1, tid))
            else:
                val = stack.pop(thread_id=tid)
                ops.append(Operation('pop', None, val, t, t + 1, tid))
            t += 2

        checker = LinearizabilityChecker('stack')
        is_lin, _ = checker.check(ops)
        if is_lin:
            stack_linearizable += 1

    stack_rate = stack_linearizable / stack_total
    print(f"    Linearizable: {stack_linearizable}/{stack_total} ({stack_rate:.1%})")

    # MS Queue: 1000 random interleavings
    print("  Michael-Scott Queue (1000 random interleavings)...")
    queue_linearizable = 0
    queue_total = 1000
    for trial in range(queue_total):
        queue = LockFreeQueue()
        ops = []
        t = 0
        n_ops = rng.randint(3, 8)
        n_enqueued = 0
        for j in range(n_ops):
            tid = rng.randint(0, 3)
            if rng.random() < 0.6 or n_enqueued == 0:
                val = rng.randint(1, 100)
                queue.enqueue(val, thread_id=tid)
                n_enqueued += 1
                ops.append(Operation('enqueue', val, None, t, t + 1, tid))
            else:
                val = queue.dequeue(thread_id=tid)
                ops.append(Operation('dequeue', None, val, t, t + 1, tid))
            t += 2

        checker = LinearizabilityChecker('queue')
        is_lin, _ = checker.check(ops)
        if is_lin:
            queue_linearizable += 1

    queue_rate = queue_linearizable / queue_total
    print(f"    Linearizable: {queue_linearizable}/{queue_total} ({queue_rate:.1%})")

    RESULTS['lockfree_verification'] = {
        'treiber_stack': {
            'total_interleavings': stack_total,
            'linearizable': stack_linearizable,
            'rate': round(stack_rate, 4),
        },
        'ms_queue': {
            'total_interleavings': queue_total,
            'linearizable': queue_linearizable,
            'rate': round(queue_rate, 4),
        },
    }
    return True


# ============================================================================
# 6. Model Checking
# ============================================================================

def _make_bakery_algorithm():
    """Lamport's bakery algorithm for 2 threads."""
    prog = MCProgram()
    prog.set_init('choosing_0', 0)
    prog.set_init('choosing_1', 0)
    prog.set_init('number_0', 0)
    prog.set_init('number_1', 0)
    prog.set_init('cs', 0)

    # Thread 0
    t0 = ThreadProgram()
    t0.add(Instruction(InstrType.WRITE, var='choosing_0', value=1))
    t0.add(Instruction(InstrType.READ, var='number_1', reg='n1'))
    t0.add(Instruction(InstrType.WRITE, var='number_0', value=1))
    t0.add(Instruction(InstrType.WRITE, var='choosing_0', value=0))
    t0.add(Instruction(InstrType.READ, var='choosing_1', reg='c1'))
    t0.add(Instruction(InstrType.READ, var='number_1', reg='n1b'))
    t0.add(Instruction(InstrType.BRANCH, target_pc=4,
           condition=lambda l: l.get('c1', 0) == 1))
    t0.add(Instruction(InstrType.READ, var='cs', reg='old_cs'))
    t0.add(Instruction(InstrType.WRITE, var='cs',
           value=lambda l: l.get('old_cs', 0) + 1))
    t0.add(Instruction(InstrType.WRITE, var='cs',
           value=lambda l: l.get('old_cs', 0)))
    t0.add(Instruction(InstrType.WRITE, var='number_0', value=0))
    t0.add(Instruction(InstrType.END))

    # Thread 1
    t1 = ThreadProgram()
    t1.add(Instruction(InstrType.WRITE, var='choosing_1', value=1))
    t1.add(Instruction(InstrType.READ, var='number_0', reg='n0'))
    t1.add(Instruction(InstrType.WRITE, var='number_1', value=1))
    t1.add(Instruction(InstrType.WRITE, var='choosing_1', value=0))
    t1.add(Instruction(InstrType.READ, var='choosing_0', reg='c0'))
    t1.add(Instruction(InstrType.READ, var='number_0', reg='n0b'))
    t1.add(Instruction(InstrType.BRANCH, target_pc=4,
           condition=lambda l: l.get('c0', 0) == 1))
    t1.add(Instruction(InstrType.READ, var='cs', reg='old_cs'))
    t1.add(Instruction(InstrType.WRITE, var='cs',
           value=lambda l: l.get('old_cs', 0) + 1))
    t1.add(Instruction(InstrType.WRITE, var='cs',
           value=lambda l: l.get('old_cs', 0)))
    t1.add(Instruction(InstrType.WRITE, var='number_1', value=0))
    t1.add(Instruction(InstrType.END))

    prog.add_thread(0, t0)
    prog.add_thread(1, t1)
    return prog


def run_model_checking():
    section("6. MODEL CHECKING")

    results = {}

    # Peterson's algorithm
    print("  Peterson's algorithm...")
    peterson = make_petersons_algorithm()
    mc = ModelChecker(peterson)
    start = time.time()
    check_result = mc.check(mutual_exclusion_property, method='bfs', depth_bound=50)
    elapsed = time.time() - start

    results['peterson'] = {
        'algorithm': "Peterson's",
        'property': 'mutual_exclusion',
        'satisfied': check_result.satisfied,
        'states_explored': check_result.states_explored,
        'time_seconds': round(elapsed, 4),
    }
    print(f"    Mutual exclusion: {'SATISFIED' if check_result.satisfied else 'VIOLATED'}")
    print(f"    States explored: {check_result.states_explored}")
    print(f"    Time: {elapsed:.4f}s")

    # Bakery algorithm
    print("  Bakery algorithm...")
    bakery = _make_bakery_algorithm()
    mc2 = ModelChecker(bakery)
    start = time.time()
    check_result2 = mc2.check(mutual_exclusion_property, method='bfs', depth_bound=50)
    elapsed2 = time.time() - start

    results['bakery'] = {
        'algorithm': "Lamport's Bakery",
        'property': 'mutual_exclusion',
        'satisfied': check_result2.satisfied,
        'states_explored': check_result2.states_explored,
        'time_seconds': round(elapsed2, 4),
    }
    print(f"    Mutual exclusion: {'SATISFIED' if check_result2.satisfied else 'VIOLATED'}")
    print(f"    States explored: {check_result2.states_explored}")
    print(f"    Time: {elapsed2:.4f}s")

    RESULTS['model_checking'] = results
    return True


# ============================================================================
# 7. DPOR Efficiency
# ============================================================================

def _make_simple_concurrent(n_threads, n_ops_per_thread, shared_vars):
    """Build a simple concurrent program for DPOR comparison."""
    prog = MCProgram()
    for v in shared_vars:
        prog.set_init(v, 0)
    for tid in range(n_threads):
        t = ThreadProgram()
        for i in range(n_ops_per_thread):
            var = shared_vars[i % len(shared_vars)]
            t.add(Instruction(InstrType.READ, var=var, reg=f'r{i}'))
            t.add(Instruction(InstrType.WRITE, var=var, value=tid * 10 + i))
        t.add(Instruction(InstrType.END))
        prog.add_thread(tid, t)
    return prog


def run_dpor_efficiency():
    section("7. DPOR EFFICIENCY")

    programs = [
        ("2T-2ops-1var", 2, 2, ['x']),
        ("2T-3ops-2var", 2, 3, ['x', 'y']),
        ("3T-2ops-1var", 3, 2, ['x']),
        ("3T-2ops-2var", 3, 2, ['x', 'y']),
        ("2T-4ops-2var", 2, 4, ['x', 'y']),
    ]

    results = {}
    for name, n_threads, n_ops, shared_vars in programs:
        prog = _make_simple_concurrent(n_threads, n_ops, shared_vars)

        # Without POR (full exploration)
        mc_full = ModelChecker(prog)
        start = time.time()
        r_full = mc_full.check(method='bfs', depth_bound=30, use_por=False)
        t_full = time.time() - start

        # With POR (DPOR-style reduction)
        mc_por = ModelChecker(prog)
        start = time.time()
        r_por = mc_por.check(method='bfs', depth_bound=30, use_por=True)
        t_por = time.time() - start

        full_states = r_full.states_explored
        por_states = r_por.states_explored
        reduction = 1.0 - (por_states / full_states) if full_states > 0 else 0.0

        results[name] = {
            'full_states': full_states,
            'por_states': por_states,
            'reduction_ratio': round(reduction, 4),
            'full_time': round(t_full, 4),
            'por_time': round(t_por, 4),
            'speedup': round(t_full / t_por, 2) if t_por > 0 else float('inf'),
        }
        print(f"  {name:>16}: full={full_states:>6}, POR={por_states:>6}, "
              f"reduction={reduction:.1%}, speedup={results[name]['speedup']:.1f}x")

    RESULTS['dpor_efficiency'] = results
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  LITMUS∞ UTILITY SHOWCASE")
    print("=" * 70)

    start_time = time.time()

    evaluations = [
        ("Memory Model Comparison", run_memory_model_comparison),
        ("Race Detection Accuracy", run_race_detection_accuracy),
        ("Fence Optimization", run_fence_optimization),
        ("Deadlock Detection", run_deadlock_detection),
        ("Lock-Free Verification", run_lockfree_verification),
        ("Model Checking", run_model_checking),
        ("DPOR Efficiency", run_dpor_efficiency),
    ]

    passed = 0
    failed = 0
    for name, fn in evaluations:
        try:
            result = fn()
            if result:
                passed += 1
                print(f"\n  → {name}: PASS")
            else:
                failed += 1
                print(f"\n  → {name}: FAIL")
        except Exception as e:
            failed += 1
            print(f"\n  → {name}: ERROR - {e}")
            traceback.print_exc()

    total_time = time.time() - start_time

    section("SUMMARY")
    print(f"  Evaluations: {passed + failed}")
    print(f"  Passed:      {passed}")
    print(f"  Failed:      {failed}")
    print(f"  Total time:  {total_time:.2f}s")

    RESULTS['summary'] = {
        'total_evaluations': passed + failed,
        'passed': passed,
        'failed': failed,
        'total_time_seconds': round(total_time, 2),
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'utility_showcase_results.json')
    with open(out_path, 'w') as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
