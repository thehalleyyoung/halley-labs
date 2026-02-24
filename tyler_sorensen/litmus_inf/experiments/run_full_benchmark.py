"""
Full benchmark suite for the concurrency platform.

Tests all memory models, race detectors, fence optimizer, deadlock detector,
concurrent data structures, and model checker. Produces litmus_benchmark_results.json.
"""

import sys
import os
import json
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from memory_model import (
    MemoryEvent, ExecutionBuilder, FenceType, Scope,
    SequentialConsistency, TotalStoreOrder, PartialStoreOrder,
    RelaxedMemoryModel, RISCVModel, PTXModel, VulkanModel,
    ConsistencyChecker, get_model,
)
from litmus_test_engine import (
    make_store_buffering, make_message_passing, make_load_buffering,
    make_iriw, make_wrc, make_corr,
    generate_all_outcomes, check_outcome, classify_test,
    litmus_test_from_string, LitmusTestSuite, RandomTestGenerator,
    Outcome,
)
from race_detector import (
    HappensBeforeDetector, EraserDetector, FastTrackDetector,
    TSanDetector, HybridDetector, TraceReplayer,
    TraceEvent, make_trace_event,
    generate_racy_trace, generate_synchronized_trace,
    generate_complex_race_trace,
)
from fence_optimizer import (
    FenceOptimizationPipeline, FenceVerifier,
    make_store_buffering_program, make_message_passing_program,
    make_dekker_program, ConcurrentProgram,
)
from deadlock_detector import (
    DeadlockDetector, LivelockDetector, PriorityInversionDetector,
    LockEvent,
)
from concurrent_data_structures import (
    LockFreeStack, LockFreeQueue, ReadWriteLock,
    ConcurrentHashMap, SkipList,
    LinearizabilityChecker, Operation,
)
from model_checker import (
    ModelChecker, BoundedModelChecker, CTLModelChecker,
    make_petersons_algorithm, mutual_exclusion_property,
    make_racy_counter,
)


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

class BenchmarkResult:
    def __init__(self, name, passed, details=None, elapsed=0.0):
        self.name = name
        self.passed = passed
        self.details = details or {}
        self.elapsed = elapsed

    def to_dict(self):
        return {
            'name': self.name,
            'passed': self.passed,
            'details': self.details,
            'elapsed_seconds': round(self.elapsed, 4),
        }


def run_benchmark(name, fn):
    """Run a single benchmark and return BenchmarkResult."""
    start = time.time()
    try:
        passed, details = fn()
        elapsed = time.time() - start
        return BenchmarkResult(name, passed, details, elapsed)
    except Exception as e:
        elapsed = time.time() - start
        return BenchmarkResult(name, False, {'error': str(e)}, elapsed)


# ---------------------------------------------------------------------------
# 1. Memory Model Benchmarks
# ---------------------------------------------------------------------------

def benchmark_memory_models_sb():
    """Test all memory models on Store Buffering (SB)."""
    MemoryEvent._counter = 0
    # SB: T0: W(x,1); R(y,0)  T1: W(y,1); R(x,0)
    # Forbidden outcome (both read 0) under SC, allowed under TSO
    eb = ExecutionBuilder()
    eb.init(0, 0)
    eb.init(1, 0)
    w0 = eb.write(0, 0, 1)
    r0 = eb.read(0, 1, 0)
    w1 = eb.write(1, 1, 1)
    r1 = eb.read(1, 0, 0)
    iw0 = eb.execution.init_writes[0]
    iw1 = eb.execution.init_writes[1]
    eb.rf(iw1, r0)
    eb.rf(iw0, r1)
    eb.co(iw0, w0)
    eb.co(iw1, w1)
    exe = eb.build()

    results = {}
    models = {
        'SC': SequentialConsistency(),
        'TSO': TotalStoreOrder(),
        'PSO': PartialStoreOrder(),
        'Relaxed': RelaxedMemoryModel(),
        'RVWMO': RISCVModel(),
    }

    expected = {
        'SC': 'forbidden',
        'TSO': 'allowed',
        'PSO': 'allowed',
        'Relaxed': 'allowed',
        'RVWMO': 'allowed',
    }

    all_correct = True
    for name, model in models.items():
        result, reason = model.check(exe)
        correct = result == expected[name]
        results[name] = {'result': result, 'expected': expected[name], 'correct': correct}
        if not correct:
            all_correct = False

    return all_correct, {'test': 'SB', 'models': results}


def benchmark_memory_models_mp():
    """Test all models on Message Passing (MP)."""
    MemoryEvent._counter = 0
    # MP: T0: W(x,1); W(y,1)  T1: R(y,1); R(x,0)
    # Forbidden under SC (if you see y=1 you must see x=1)
    eb = ExecutionBuilder()
    eb.init(0, 0)
    eb.init(1, 0)
    w0x = eb.write(0, 0, 1)
    w0y = eb.write(0, 1, 1)
    r1y = eb.read(1, 1, 1)
    r1x = eb.read(1, 0, 0)
    iw0 = eb.execution.init_writes[0]
    iw1 = eb.execution.init_writes[1]
    eb.rf(w0y, r1y)   # T1 reads y=1 from T0's write
    eb.rf(iw0, r1x)   # T1 reads x=0 from init
    eb.co(iw0, w0x)
    eb.co(iw1, w0y)
    exe = eb.build()

    results = {}
    sc = SequentialConsistency()
    tso = TotalStoreOrder()

    r_sc, _ = sc.check(exe)
    r_tso, _ = tso.check(exe)

    results['SC'] = r_sc
    results['TSO'] = r_tso

    # SC should forbid this, TSO should also forbid (TSO preserves store-store order)
    correct = r_sc == 'forbidden'
    return correct, {'test': 'MP', 'results': results}


def benchmark_memory_models_lb():
    """Test Load Buffering (LB)."""
    MemoryEvent._counter = 0
    # LB: T0: R(x,1); W(y,1)  T1: R(y,1); W(x,1)
    # Both read 1 from each other's write - only possible with load-load/load-store reordering
    eb = ExecutionBuilder()
    eb.init(0, 0)
    eb.init(1, 0)
    r0 = eb.read(0, 0, 1)
    w0 = eb.write(0, 1, 1)
    r1 = eb.read(1, 1, 1)
    w1 = eb.write(1, 0, 1)
    iw0 = eb.execution.init_writes[0]
    iw1 = eb.execution.init_writes[1]
    eb.rf(w1, r0)   # T0 reads x=1 from T1's write
    eb.rf(w0, r1)   # T1 reads y=1 from T0's write
    eb.co(iw0, w1)
    eb.co(iw1, w0)
    exe = eb.build()

    sc = SequentialConsistency()
    tso = TotalStoreOrder()
    relaxed = RelaxedMemoryModel()

    r_sc, _ = sc.check(exe)
    r_tso, _ = tso.check(exe)
    r_rel, _ = relaxed.check(exe)

    # LB forbidden under SC and TSO, may be allowed under relaxed
    correct = r_sc == 'forbidden' and r_tso == 'forbidden'
    return correct, {'test': 'LB', 'SC': r_sc, 'TSO': r_tso, 'Relaxed': r_rel}


def benchmark_memory_models_iriw():
    """Test IRIW (Independent Reads of Independent Writes)."""
    MemoryEvent._counter = 0
    # IRIW needs 4 threads, tests multi-copy atomicity
    eb = ExecutionBuilder()
    eb.init(0, 0)
    eb.init(1, 0)
    w0 = eb.write(0, 0, 1)  # T0: W(x)=1
    w1 = eb.write(1, 1, 1)  # T1: W(y)=1
    r2x = eb.read(2, 0, 1)  # T2: R(x)=1
    r2y = eb.read(2, 1, 0)  # T2: R(y)=0
    r3y = eb.read(3, 1, 1)  # T3: R(y)=1
    r3x = eb.read(3, 0, 0)  # T3: R(x)=0

    iw0 = eb.execution.init_writes[0]
    iw1 = eb.execution.init_writes[1]

    eb.rf(w0, r2x)
    eb.rf(iw1, r2y)
    eb.rf(w1, r3y)
    eb.rf(iw0, r3x)
    eb.co(iw0, w0)
    eb.co(iw1, w1)
    exe = eb.build()

    sc = SequentialConsistency()
    r_sc, _ = sc.check(exe)

    correct = r_sc == 'forbidden'
    return correct, {'test': 'IRIW', 'SC': r_sc}


# ---------------------------------------------------------------------------
# 2. Race Detector Benchmarks
# ---------------------------------------------------------------------------

def benchmark_race_detector_racy():
    """Test race detectors on trace with known races."""
    detectors = {
        'HB': lambda: HappensBeforeDetector(n_threads=4),
        'FastTrack': lambda: FastTrackDetector(n_threads=4),
        'TSan': lambda: TSanDetector(n_threads=4),
    }

    results = {}
    all_correct = True

    for name, make_det in detectors.items():
        TraceEvent._counter = 0
        det = make_det()
        trace = generate_racy_trace()
        for e in trace:
            det.process_event(e)
        races = det.get_races()
        found = len(races) > 0
        results[name] = {'races_found': len(races), 'correct': found}
        if not found:
            all_correct = False

    return all_correct, {'test': 'racy_trace', 'detectors': results}


def benchmark_race_detector_synchronized():
    """Test race detectors on properly synchronized trace (no races expected)."""
    detectors = {
        'HB': lambda: HappensBeforeDetector(n_threads=4),
        'FastTrack': lambda: FastTrackDetector(n_threads=4),
        'TSan': lambda: TSanDetector(n_threads=4),
    }

    results = {}
    all_correct = True

    for name, make_det in detectors.items():
        TraceEvent._counter = 0
        det = make_det()
        trace = generate_synchronized_trace()
        for e in trace:
            det.process_event(e)
        races = det.get_races()
        no_races = len(races) == 0
        results[name] = {'races_found': len(races), 'correct': no_races}
        if not no_races:
            all_correct = False

    return all_correct, {'test': 'synchronized_trace', 'detectors': results}


def benchmark_race_detector_complex():
    """Test race detectors on complex trace with mixed sync/racy accesses."""
    TraceEvent._counter = 0
    det = HappensBeforeDetector(n_threads=4)
    trace = generate_complex_race_trace()
    for e in trace:
        det.process_event(e)
    races = det.get_races()
    # Should find races on y and z but not on x (which is protected by lock)
    race_addrs = {r.address for r in races}
    found_y = 'y' in race_addrs
    found_z = 'z' in race_addrs
    no_x_race = 'x' not in race_addrs

    correct = found_y and found_z and no_x_race
    return correct, {
        'test': 'complex_trace',
        'races_found': len(races),
        'race_addresses': list(race_addrs),
        'y_race': found_y,
        'z_race': found_z,
        'x_no_race': no_x_race,
    }


# ---------------------------------------------------------------------------
# 3. Fence Optimizer Benchmarks
# ---------------------------------------------------------------------------

def benchmark_fence_optimizer_sb():
    """Fence optimizer on Store Buffering for TSO."""
    prog = make_store_buffering_program()
    pipeline = FenceOptimizationPipeline()
    result = pipeline.run(prog, 'tso')

    correct = result['verified'] and result['n_fences'] > 0
    return correct, {
        'test': 'SB_TSO',
        'n_fences': result['n_fences'],
        'cost': result['reduced_cost'],
        'verified': result['verified'],
    }


def benchmark_fence_optimizer_mp_arm():
    """Fence optimizer on Message Passing for ARM."""
    prog = make_message_passing_program()
    pipeline = FenceOptimizationPipeline()
    result = pipeline.run(prog, 'arm')

    correct = result['verified'] and result['n_fences'] > 0
    return correct, {
        'test': 'MP_ARM',
        'n_fences': result['n_fences'],
        'cost': result['reduced_cost'],
        'verified': result['verified'],
    }


def benchmark_fence_optimizer_dekker():
    """Fence optimizer on Dekker's algorithm."""
    prog = make_dekker_program()
    pipeline = FenceOptimizationPipeline()
    result = pipeline.run(prog, 'tso')

    correct = result['verified'] and result['n_fences'] > 0
    return correct, {
        'test': 'Dekker_TSO',
        'n_fences': result['n_fences'],
        'cost': result['reduced_cost'],
        'verified': result['verified'],
    }


# ---------------------------------------------------------------------------
# 4. Deadlock Detector Benchmarks
# ---------------------------------------------------------------------------

def benchmark_deadlock_ab_ba():
    """Deadlock detection: AB-BA pattern."""
    LockEvent._counter = 0
    det = DeadlockDetector()
    det.add_lock_event(0, 'A', 'acquire', 1)
    det.add_lock_event(0, 'B', 'acquire', 2)
    det.add_lock_event(0, 'B', 'release', 3)
    det.add_lock_event(0, 'A', 'release', 4)
    det.add_lock_event(1, 'B', 'acquire', 5)
    det.add_lock_event(1, 'A', 'acquire', 6)
    det.add_lock_event(1, 'A', 'release', 7)
    det.add_lock_event(1, 'B', 'release', 8)
    deadlocks = det.check()
    correct = len(deadlocks) > 0
    return correct, {
        'test': 'AB_BA',
        'deadlocks_found': len(deadlocks),
        'correct': correct,
    }


def benchmark_deadlock_no_deadlock():
    """No deadlock: consistent lock ordering."""
    LockEvent._counter = 0
    det = DeadlockDetector()
    det.add_lock_event(0, 'A', 'acquire', 1)
    det.add_lock_event(0, 'B', 'acquire', 2)
    det.add_lock_event(0, 'B', 'release', 3)
    det.add_lock_event(0, 'A', 'release', 4)
    det.add_lock_event(1, 'A', 'acquire', 5)
    det.add_lock_event(1, 'B', 'acquire', 6)
    det.add_lock_event(1, 'B', 'release', 7)
    det.add_lock_event(1, 'A', 'release', 8)
    deadlocks = det.check()
    correct = len(deadlocks) == 0
    return correct, {
        'test': 'consistent_ordering',
        'deadlocks_found': len(deadlocks),
        'correct': correct,
    }


def benchmark_deadlock_three_way():
    """Three-way deadlock: A->B, B->C, C->A."""
    LockEvent._counter = 0
    det = DeadlockDetector()
    det.add_lock_event(0, 'A', 'acquire', 1)
    det.add_lock_event(0, 'B', 'acquire', 2)
    det.add_lock_event(0, 'B', 'release', 3)
    det.add_lock_event(0, 'A', 'release', 4)

    det.add_lock_event(1, 'B', 'acquire', 5)
    det.add_lock_event(1, 'C', 'acquire', 6)
    det.add_lock_event(1, 'C', 'release', 7)
    det.add_lock_event(1, 'B', 'release', 8)

    det.add_lock_event(2, 'C', 'acquire', 9)
    det.add_lock_event(2, 'A', 'acquire', 10)
    det.add_lock_event(2, 'A', 'release', 11)
    det.add_lock_event(2, 'C', 'release', 12)

    deadlocks = det.check()
    correct = len(deadlocks) > 0
    return correct, {
        'test': 'three_way_deadlock',
        'deadlocks_found': len(deadlocks),
        'correct': correct,
    }


# ---------------------------------------------------------------------------
# 5. Concurrent Data Structures Benchmarks
# ---------------------------------------------------------------------------

def benchmark_stack_linearizability():
    """Verify linearizability of Treiber stack operations."""
    stack = LockFreeStack()
    # Sequential ops with timestamps
    ops = []
    t = 0
    for v in [1, 2, 3]:
        stack.push(v, thread_id=0)
        ops.append(Operation('push', v, None, t, t + 1, 0))
        t += 2

    results = []
    for _ in range(3):
        val = stack.pop(thread_id=0)
        results.append(val)
        ops.append(Operation('pop', None, val, t, t + 1, 0))
        t += 2

    checker = LinearizabilityChecker('stack')
    is_lin, _ = checker.check(ops)

    correct = is_lin and results == [3, 2, 1]
    return correct, {
        'test': 'stack_linearizability',
        'linearizable': is_lin,
        'lifo_order': results == [3, 2, 1],
    }


def benchmark_queue_linearizability():
    """Verify linearizability of Michael-Scott queue operations."""
    queue = LockFreeQueue()
    ops = []
    t = 0
    for v in [10, 20, 30]:
        queue.enqueue(v, thread_id=0)
        ops.append(Operation('enqueue', v, None, t, t + 1, 0))
        t += 2

    results = []
    for _ in range(3):
        val = queue.dequeue(thread_id=0)
        results.append(val)
        ops.append(Operation('dequeue', None, val, t, t + 1, 0))
        t += 2

    checker = LinearizabilityChecker('queue')
    is_lin, _ = checker.check(ops)

    correct = is_lin and results == [10, 20, 30]
    return correct, {
        'test': 'queue_linearizability',
        'linearizable': is_lin,
        'fifo_order': results == [10, 20, 30],
    }


def benchmark_hashmap():
    """Test concurrent hash map operations."""
    hmap = ConcurrentHashMap()
    n = 200
    for i in range(n):
        hmap.put(f"k{i}", i)
    all_found = all(hmap.get(f"k{i}") == i for i in range(n))
    correct_size = hmap.size() == n

    hmap.put("k0", 999)
    updated = hmap.get("k0") == 999

    hmap.remove("k1")
    removed = hmap.get("k1") is None

    correct = all_found and correct_size and updated and removed
    return correct, {
        'test': 'concurrent_hashmap',
        'size': hmap.size(),
        'all_found': all_found,
        'updated': updated,
        'removed': removed,
    }


def benchmark_skiplist():
    """Test concurrent skip list operations."""
    sl = SkipList(seed=123)
    values = list(range(50))
    np.random.shuffle(values)
    for v in values:
        sl.insert(v, v * 10)

    all_found = all(sl.find(v) == v * 10 for v in range(50))
    items = sl.to_list()
    keys = [k for k, v in items]
    sorted_correctly = keys == sorted(keys)

    sl.delete(25)
    deleted = sl.find(25) is None

    correct = all_found and sorted_correctly and deleted
    return correct, {
        'test': 'concurrent_skiplist',
        'size': sl.size(),
        'all_found': all_found,
        'sorted': sorted_correctly,
        'deleted': deleted,
    }


# ---------------------------------------------------------------------------
# 6. Model Checker Benchmarks
# ---------------------------------------------------------------------------

def benchmark_model_checker_peterson():
    """Model check Peterson's algorithm for mutual exclusion."""
    peterson = make_petersons_algorithm()
    mc = ModelChecker(peterson)
    result = mc.check(mutual_exclusion_property, method='bfs', depth_bound=50)
    correct = result.satisfied
    return correct, {
        'test': 'peterson_mutual_exclusion',
        'satisfied': result.satisfied,
        'states_explored': result.states_explored,
        'time': round(result.time, 4),
    }


def benchmark_model_checker_racy_counter():
    """Model check racy counter (should find violation)."""
    racy = make_racy_counter()
    mc = ModelChecker(racy)

    def counter_always_2(state):
        enabled = mc.ts.enabled_threads(state)
        if not enabled:
            return state.shared.read('counter') == 2
        return True

    result = mc.check(counter_always_2, method='bfs', depth_bound=20)
    # Should find violation (counter != 2 is reachable)
    correct = not result.satisfied
    return correct, {
        'test': 'racy_counter',
        'violation_found': not result.satisfied,
        'states_explored': result.states_explored,
        'time': round(result.time, 4),
    }


def benchmark_bounded_model_checking():
    """Bounded model checking on Peterson's."""
    peterson = make_petersons_algorithm()
    bmc = BoundedModelChecker(peterson)
    result = bmc.check(mutual_exclusion_property, bound=30)
    correct = result.satisfied
    return correct, {
        'test': 'bounded_peterson',
        'satisfied': result.satisfied,
        'states_explored': result.states_explored,
        'time': round(result.time, 4),
    }


# ---------------------------------------------------------------------------
# 7. Litmus Test Engine Benchmarks
# ---------------------------------------------------------------------------

def benchmark_litmus_outcomes():
    """Generate all outcomes for standard litmus tests."""
    tests = {
        'SB': make_store_buffering(),
        'MP': make_message_passing(),
        'LB': make_load_buffering(),
    }

    # Under SC interleaving semantics, the "forbidden" outcomes are precisely
    # those that require weak-memory reorderings and thus do NOT appear.
    # This confirms the litmus tests correctly identify non-SC behaviors.
    # SB forbidden (r0=0,r1=0) NOT reachable under SC (requires store-load reorder)
    # MP forbidden (r0=1,r1=0) NOT reachable under SC (requires store-store or load-load reorder)
    # LB forbidden (r0=1,r1=1) NOT reachable under SC (requires load-store reorder)
    expect_forbidden_reachable = {'SB': False, 'MP': False, 'LB': False}

    results = {}
    all_correct = True

    for name, test in tests.items():
        outcomes = generate_all_outcomes(test)
        n_outcomes = len(outcomes)
        has_forbidden = False
        if test.forbidden_outcomes:
            fo = test.forbidden_outcomes[0]
            for out in outcomes:
                if all(out.get(k) == v for k, v in fo.values.items()):
                    has_forbidden = True
                    break

        expected = expect_forbidden_reachable[name]
        correct = has_forbidden == expected

        results[name] = {
            'n_outcomes': n_outcomes,
            'has_forbidden': has_forbidden,
            'expected': expected,
            'correct': correct,
        }
        if not correct:
            all_correct = False

    return all_correct, {'test': 'litmus_outcomes', 'results': results}


def benchmark_litmus_parser():
    """Test litmus test parser."""
    text = """
    name: SB_parsed
    init: x=0; y=0
    T0: W(x)=1; r0=R(y)
    T1: W(y)=1; r1=R(x)
    forbidden: r0=0, r1=0
    """
    parsed = litmus_test_from_string(text)
    correct = (parsed.name == "SB_parsed"
               and len(parsed.threads) == 2
               and len(parsed.forbidden_outcomes) == 1)
    return correct, {
        'test': 'litmus_parser',
        'name': parsed.name,
        'n_threads': len(parsed.threads),
        'n_forbidden': len(parsed.forbidden_outcomes),
    }


def benchmark_random_test_generation():
    """Test random litmus test generation."""
    gen = RandomTestGenerator(seed=42)
    tests = gen.generate_batch(20, n_threads=2, n_instructions=3)
    all_valid = all(len(t.threads) == 2 and
                    all(len(th) == 3 for th in t.threads)
                    for t in tests)
    return all_valid, {
        'test': 'random_generation',
        'n_tests': len(tests),
        'all_valid': all_valid,
    }


# ---------------------------------------------------------------------------
# Run all benchmarks
# ---------------------------------------------------------------------------

def run_all_benchmarks():
    """Run all benchmarks and produce results."""
    benchmarks = [
        # Memory models
        ("memory_model_SB", benchmark_memory_models_sb),
        ("memory_model_MP", benchmark_memory_models_mp),
        ("memory_model_LB", benchmark_memory_models_lb),
        ("memory_model_IRIW", benchmark_memory_models_iriw),

        # Race detectors
        ("race_detector_racy", benchmark_race_detector_racy),
        ("race_detector_synchronized", benchmark_race_detector_synchronized),
        ("race_detector_complex", benchmark_race_detector_complex),

        # Fence optimizer
        ("fence_optimizer_SB", benchmark_fence_optimizer_sb),
        ("fence_optimizer_MP_ARM", benchmark_fence_optimizer_mp_arm),
        ("fence_optimizer_Dekker", benchmark_fence_optimizer_dekker),

        # Deadlock detector
        ("deadlock_AB_BA", benchmark_deadlock_ab_ba),
        ("deadlock_no_deadlock", benchmark_deadlock_no_deadlock),
        ("deadlock_three_way", benchmark_deadlock_three_way),

        # Concurrent data structures
        ("stack_linearizability", benchmark_stack_linearizability),
        ("queue_linearizability", benchmark_queue_linearizability),
        ("hashmap_operations", benchmark_hashmap),
        ("skiplist_operations", benchmark_skiplist),

        # Model checker
        ("model_checker_peterson", benchmark_model_checker_peterson),
        ("model_checker_racy_counter", benchmark_model_checker_racy_counter),
        ("bounded_model_checking", benchmark_bounded_model_checking),

        # Litmus test engine
        ("litmus_outcomes", benchmark_litmus_outcomes),
        ("litmus_parser", benchmark_litmus_parser),
        ("random_test_generation", benchmark_random_test_generation),
    ]

    results = []
    passed = 0
    failed = 0

    print("=" * 70)
    print("  CONCURRENCY PLATFORM BENCHMARK SUITE")
    print("=" * 70)
    print()

    for name, fn in benchmarks:
        result = run_benchmark(name, fn)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        status_icon = "✓" if result.passed else "✗"
        print(f"  {status_icon} {status:4s}  {name:40s}  ({result.elapsed:.3f}s)")

        if result.passed:
            passed += 1
        else:
            failed += 1
            if 'error' in result.details:
                print(f"         Error: {result.details['error']}")

    print()
    print("-" * 70)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("-" * 70)

    # Write JSON results
    output = {
        'summary': {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': round(passed / max(total, 1), 4),
        },
        'benchmarks': [r.to_dict() for r in results],
    }

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'litmus_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to: {output_path}")

    # Detailed summary by category
    print()
    print("  Category Summary:")
    categories = {
        'Memory Models': [r for r in results if r.name.startswith('memory_model')],
        'Race Detectors': [r for r in results if r.name.startswith('race_detector')],
        'Fence Optimizer': [r for r in results if r.name.startswith('fence_optimizer')],
        'Deadlock Detector': [r for r in results if r.name.startswith('deadlock')],
        'Data Structures': [r for r in results if r.name.startswith(('stack', 'queue', 'hashmap', 'skiplist'))],
        'Model Checker': [r for r in results if r.name.startswith(('model_checker', 'bounded'))],
        'Litmus Engine': [r for r in results if r.name.startswith(('litmus', 'random'))],
    }

    for cat_name, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r.passed)
        cat_total = len(cat_results)
        status = "✓" if cat_passed == cat_total else "✗"
        print(f"    {status} {cat_name:25s}: {cat_passed}/{cat_total}")

    print()
    return failed == 0


if __name__ == "__main__":
    success = run_all_benchmarks()
    sys.exit(0 if success else 1)
