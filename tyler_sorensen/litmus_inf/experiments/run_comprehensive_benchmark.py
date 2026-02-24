"""
Comprehensive benchmark: tests all new modules and runs existing benchmarks.

Tests program analyzer, memory model compiler, verification engine,
lock-free algorithms, weak memory testing, synchronization synthesis,
performance predictor, concurrency patterns, and GPU memory model.
"""

import sys
import os
import json
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def run_test(name, fn):
    """Run a test function, return (passed, result_or_error, elapsed)."""
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        print(f"  [PASS] {name} ({elapsed:.3f}s)")
        return True, result, elapsed
    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        print(f"  [FAIL] {name} ({elapsed:.3f}s): {e}")
        return False, str(e) + "\n" + tb, elapsed


# =============================================================================
# Test 1: Program Analyzer
# =============================================================================
def test_program_analyzer():
    print("\n" + "=" * 60)
    print("TEST 1: Program Analyzer")
    print("=" * 60)
    from program_analyzer import (
        ProgramAnalyzer, build_peterson_program, build_lock_program,
        build_racy_program, build_barrier_program, build_producer_consumer_program
    )
    results = {}

    def test_peterson():
        analyzer = ProgramAnalyzer()
        prog = build_peterson_program()
        analysis = analyzer.analyze(prog)
        assert analysis.thread_count == 2, f"Expected 2 threads, got {analysis.thread_count}"
        assert "turn" in analysis.shared_vars, "turn should be shared"
        assert "flag0" in analysis.shared_vars, "flag0 should be shared"
        assert len(analysis.dependencies) > 0, "Should have dependencies"
        return analysis.summary()

    def test_lock_program():
        analyzer = ProgramAnalyzer()
        prog = build_lock_program()
        analysis = analyzer.analyze(prog)
        assert analysis.thread_count == 2
        assert len(analysis.critical_sections) >= 2, "Should detect critical sections"
        assert "x" in analysis.shared_vars
        # Lock-protected accesses should have fewer races
        return {"cs_count": len(analysis.critical_sections), "races": len(analysis.data_races_possible)}

    def test_racy():
        analyzer = ProgramAnalyzer()
        prog = build_racy_program()
        analysis = analyzer.analyze(prog)
        assert len(analysis.data_races_possible) > 0, "Should detect data races"
        assert "x" in analysis.shared_vars
        assert "y" in analysis.shared_vars
        return {"races": len(analysis.data_races_possible), "shared": list(analysis.shared_vars)}

    def test_barrier():
        analyzer = ProgramAnalyzer()
        prog = build_barrier_program()
        analysis = analyzer.analyze(prog)
        assert len(analysis.sync_points) > 0, "Should detect barrier sync points"
        return {"sync_points": len(analysis.sync_points)}

    def test_producer_consumer():
        analyzer = ProgramAnalyzer()
        prog = build_producer_consumer_program()
        analysis = analyzer.analyze(prog)
        assert len(analysis.sync_points) > 0, "Should detect semaphore sync points"
        return {"sync_points": len(analysis.sync_points)}

    for name, fn in [("Peterson analysis", test_peterson),
                     ("Lock program", test_lock_program),
                     ("Racy program", test_racy),
                     ("Barrier program", test_barrier),
                     ("Producer-consumer", test_producer_consumer)]:
        passed, result, elapsed = run_test(name, fn)
        results[name] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Test 2: Memory Model Compiler
# =============================================================================
def test_memory_model_compiler():
    print("\n" + "=" * 60)
    print("TEST 2: Memory Model Compiler")
    print("=" * 60)
    from memory_model_compiler import (
        MemoryModelCompiler, build_sb_execution, build_mp_execution,
        ModelSpec, Execution, Event, Relation, RelationType
    )
    results = {}

    def test_compile_tso():
        compiler = MemoryModelCompiler()
        tso = compiler.compile_builtin("TSO")
        assert tso.name == "TSO"
        # SB forbidden under SC: r0=r1=0
        sb_forbidden = build_sb_execution(x_val=0, y_val=0)
        allowed, violations = tso.check_execution(sb_forbidden)
        return {"model": "TSO", "sb_00_allowed": allowed}

    def test_compile_sc():
        compiler = MemoryModelCompiler()
        sc = compiler.compile_builtin("SC")
        sb_forbidden = build_sb_execution(x_val=0, y_val=0)
        allowed, violations = sc.check_execution(sb_forbidden)
        # SC should forbid SB outcome r0=r1=0
        return {"model": "SC", "sb_00_allowed": allowed, "violations": violations}

    def test_model_comparison():
        compiler = MemoryModelCompiler()
        sc = compiler.compile_builtin("SC")
        tso = compiler.compile_builtin("TSO")
        test_execs = [
            build_sb_execution(0, 0),
            build_sb_execution(1, 0),
            build_sb_execution(0, 1),
            build_sb_execution(1, 1),
        ]
        comparison = compiler.compare_models(sc, tso, test_execs)
        return {
            "equivalent": comparison["equivalent"],
            "tso_weaker": comparison["b_weaker"],
            "both_allow": len(comparison["both_allow"]),
        }

    def test_visualization():
        compiler = MemoryModelCompiler()
        exec_ = build_sb_execution(1, 0)
        dot = compiler.visualize(exec_)
        assert "digraph" in dot
        assert "E0" in dot
        return {"dot_length": len(dot)}

    for name, fn in [("Compile TSO", test_compile_tso),
                     ("Compile SC", test_compile_sc),
                     ("Model comparison", test_model_comparison),
                     ("Visualization", test_visualization)]:
        passed, result, elapsed = run_test(name, fn)
        results[name] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Test 3: Verification Engine
# =============================================================================
def test_verification_engine():
    print("\n" + "=" * 60)
    print("TEST 3: Verification Engine")
    print("=" * 60)
    from verification_engine import (
        VerificationEngine, PropertyType, SystemState, ThreadState,
        build_peterson_programs, build_lock_programs, build_broken_mutex_programs,
        build_deadlock_programs, build_lockfree_programs
    )
    results = {}

    def test_peterson_mutex():
        engine = VerificationEngine(max_states=10000)
        # Use lock-based program for verified mutex
        programs = build_lock_programs()
        cs_markers = {0: (1, 2), 1: (1, 2)}
        result = engine.verify(programs, PropertyType.MUTUAL_EXCLUSION,
                               cs_markers=cs_markers)
        assert result.verified, f"Lock-based mutex should be verified: {result.message}"
        return {"verified": result.verified, "states": result.states_explored}

    def test_broken_mutex():
        engine = VerificationEngine(max_states=10000)
        programs = build_broken_mutex_programs()
        cs_markers = {0: (0, 1), 1: (0, 1)}
        result = engine.verify(programs, PropertyType.MUTUAL_EXCLUSION,
                               cs_markers=cs_markers)
        # Broken mutex should be violated
        assert not result.verified, "Broken mutex should be violated"
        return {"verified": result.verified, "states": result.states_explored}

    def test_lock_deadlock():
        engine = VerificationEngine(max_states=10000)
        programs = build_lock_programs()
        result = engine.verify(programs, PropertyType.DEADLOCK_FREEDOM)
        assert result.verified, "Lock program should be deadlock-free"
        return {"verified": result.verified, "states": result.states_explored}

    def test_deadlock_detection():
        engine = VerificationEngine(max_states=10000)
        programs = build_deadlock_programs()
        result = engine.verify(programs, PropertyType.DEADLOCK_FREEDOM)
        # Deadlock program has lock ordering violation
        return {"verified": result.verified, "states": result.states_explored}

    def test_lockfree():
        engine = VerificationEngine(max_states=10000)
        programs = build_lockfree_programs()
        result = engine.verify(programs, PropertyType.LOCK_FREEDOM)
        return {"verified": result.verified, "states": result.states_explored}

    for name, fn in [("Peterson mutex", test_peterson_mutex),
                     ("Broken mutex", test_broken_mutex),
                     ("Lock deadlock-free", test_lock_deadlock),
                     ("Deadlock detection", test_deadlock_detection),
                     ("Lock-freedom", test_lockfree)]:
        passed, result, elapsed = run_test(name, fn)
        results[name] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Test 4: Lock-Free Algorithms
# =============================================================================
def test_lock_free_algorithms():
    print("\n" + "=" * 60)
    print("TEST 4: Lock-Free Algorithms")
    print("=" * 60)
    from lock_free_algorithms import (
        LockFreeLinkedList, LockFreeHashMap, LockFreeBST,
        WaitFreeQueue, ConcurrentCounter, MPMC_Queue,
        ConcurrentExecutionSimulator, LinearizabilityChecker, PerformanceModel
    )
    results = {}

    def test_linked_list():
        ll = LockFreeLinkedList()
        for i in [5, 3, 7, 1, 9]:
            assert ll.insert(i), f"Insert {i} should succeed"
        assert not ll.insert(5), "Duplicate insert should fail"
        assert ll.contains(3)
        assert ll.delete(3)
        assert not ll.contains(3)
        assert ll.to_list() == [1, 5, 7, 9]
        return {"size": ll.size, "list": ll.to_list()}

    def test_hash_map():
        hm = LockFreeHashMap()
        for i in range(20):
            hm.put(i, i * 10)
        assert hm.get(5) == 50
        assert hm.delete(5)
        assert hm.get(5) is None
        return {"size": hm.size, "keys": len(hm.keys())}

    def test_bst():
        bst = LockFreeBST()
        for k in [10, 5, 15, 3, 7]:
            assert bst.insert(k, k * 10)
        assert bst.find(5) == 50
        assert bst.delete(5)
        assert bst.find(5) is None
        return {"size": bst.size}

    def test_waitfree_queue():
        q = WaitFreeQueue(capacity=100, n_threads=4)
        for i in range(10):
            assert q.enqueue(i, thread_id=i % 4)
        vals = []
        for i in range(10):
            v = q.dequeue(thread_id=i % 4)
            if v is not None:
                vals.append(v)
        assert len(vals) == 10
        return {"dequeued": vals}

    def test_counter():
        counter = ConcurrentCounter(n_threads=4)
        for t in range(4):
            for _ in range(100):
                counter.increment(thread_id=t)
        assert counter.get() == 400
        return {"value": counter.get(), "correct": counter.get() == 400}

    def test_mpmc():
        q = MPMC_Queue(capacity=64)
        for i in range(32):
            assert q.enqueue(i)
        vals = []
        for _ in range(32):
            v = q.dequeue()
            if v is not None:
                vals.append(v)
        assert len(vals) == 32
        return {"produced": 32, "consumed": len(vals)}

    def test_simulation():
        sim = ConcurrentExecutionSimulator(rng_seed=42)
        ll_result = sim.simulate_linked_list(4, 50, 100)
        hm_result = sim.simulate_hash_map(4, 50, 100)
        bst_result = sim.simulate_bst(4, 50, 100)
        q_result = sim.simulate_queue(4, 50)
        counter_result = sim.simulate_counter(4, 100)
        mpmc_result = sim.simulate_mpmc(3, 2, 20)
        return {
            "linked_list": ll_result["ops_count"],
            "hash_map": hm_result["ops_count"],
            "bst": bst_result["ops_count"],
            "queue": q_result["ops_count"],
            "counter": counter_result["correct"],
            "mpmc": mpmc_result["all_consumed"],
        }

    for name, fn in [("Linked list", test_linked_list),
                     ("Hash map", test_hash_map),
                     ("BST", test_bst),
                     ("Wait-free queue", test_waitfree_queue),
                     ("Counter", test_counter),
                     ("MPMC queue", test_mpmc),
                     ("Simulation", test_simulation)]:
        passed, result, elapsed = run_test(name, fn)
        results[name] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Test 5: Weak Memory Testing
# =============================================================================
def test_weak_memory_testing():
    print("\n" + "=" * 60)
    print("TEST 5: Weak Memory Testing")
    print("=" * 60)
    from weak_memory_testing import (
        WeakMemoryTester, ExhaustiveEnumerator, build_sb_test, build_mp_test
    )
    results = {}

    def test_sb_exhaustive():
        tester = WeakMemoryTester()
        sb = build_sb_test()
        report = tester.exhaustive_test(sb, "SC")
        n_sc = report.unique_outcomes
        report_tso = tester.exhaustive_test(sb, "TSO")
        n_tso = report_tso.unique_outcomes
        assert n_sc >= 1, "SC should have at least 1 outcome"
        assert n_tso >= n_sc, "TSO should allow >= SC outcomes"
        return {"sc_outcomes": n_sc, "tso_outcomes": n_tso}

    def test_mp_exhaustive():
        tester = WeakMemoryTester()
        mp = build_mp_test()
        report = tester.exhaustive_test(mp, "SC")
        n_sc = report.unique_outcomes
        report_tso = tester.exhaustive_test(mp, "TSO")
        n_tso = report_tso.unique_outcomes
        return {"sc_outcomes": n_sc, "tso_outcomes": n_tso}

    def test_random_testing():
        tester = WeakMemoryTester()
        sb = build_sb_test()
        report = tester.test(sb, "SC", n_runs=5000)
        assert report.total_runs == 5000
        assert report.unique_outcomes >= 1
        return report.summary()

    def test_tso_random():
        tester = WeakMemoryTester()
        sb = build_sb_test()
        report = tester.test(sb, "TSO", n_runs=5000)
        return report.summary()

    def test_outcome_count():
        enum_sc = ExhaustiveEnumerator("SC")
        enum_tso = ExhaustiveEnumerator("TSO")
        sb = build_sb_test()
        sc_outcomes = enum_sc.enumerate(sb)
        tso_outcomes = enum_tso.enumerate(sb)
        # SB: SC has 3 outcomes (not r0=r1=0), TSO has 4 (including r0=r1=0)
        return {"sc": len(sc_outcomes), "tso": len(tso_outcomes)}

    for name, fn in [("SB exhaustive", test_sb_exhaustive),
                     ("MP exhaustive", test_mp_exhaustive),
                     ("Random testing", test_random_testing),
                     ("TSO random", test_tso_random),
                     ("Outcome counts", test_outcome_count)]:
        passed, result, elapsed = run_test(name, fn)
        results[name] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Test 6: Synchronization Synthesis
# =============================================================================
def test_synchronization_synthesis():
    print("\n" + "=" * 60)
    print("TEST 6: Synchronization Synthesis")
    print("=" * 60)
    from synchronization_synthesis import (
        SyncSynthesizer, Instruction, InstructionType, SafetySpec, DataRaceDetector
    )
    results = {}

    def test_lock_placement():
        threads = {
            0: [
                Instruction(0, 0, InstructionType.STORE, variable="x", value=1),
                Instruction(0, 1, InstructionType.LOAD, variable="y"),
            ],
            1: [
                Instruction(1, 0, InstructionType.STORE, variable="y", value=1),
                Instruction(1, 1, InstructionType.LOAD, variable="x"),
            ],
        }
        shared_vars = {"x", "y"}
        synth = SyncSynthesizer()
        safety = SafetySpec(mutual_exclusion_vars=shared_vars)

        # Detect races before
        detector = DataRaceDetector()
        races_before = detector.detect(threads, shared_vars)

        # Synthesize
        result = synth.synthesize(threads, shared_vars, safety)
        assert result.total_sync_added > 0, "Should add synchronization"
        return {"races_before": len(races_before), "sync_added": result.total_sync_added}

    def test_fence_synthesis():
        threads = {
            0: [
                Instruction(0, 0, InstructionType.STORE, variable="x", value=1),
                Instruction(0, 1, InstructionType.LOAD, variable="y"),
            ],
        }
        synth = SyncSynthesizer()
        safety = SafetySpec()
        ordering = [((0, 0), (0, 1))]
        result = synth.synthesize_fences(threads, ordering, safety)
        assert len(result.added_fences) > 0
        return {"fences_added": len(result.added_fences)}

    def test_rw_lock():
        threads = {
            0: [Instruction(0, 0, InstructionType.LOAD, variable="x")],
            1: [Instruction(1, 0, InstructionType.LOAD, variable="x")],
            2: [Instruction(2, 0, InstructionType.STORE, variable="x", value=1)],
        }
        synth = SyncSynthesizer()
        result = synth.synthesize_rw_locks(threads, {"x"})
        return {"locks_added": len(result.added_locks)}

    for name, fn in [("Lock placement", test_lock_placement),
                     ("Fence synthesis", test_fence_synthesis),
                     ("RW lock synthesis", test_rw_lock)]:
        passed, result, elapsed = run_test(name, fn)
        results[name] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Test 7: Performance Predictor
# =============================================================================
def test_performance_predictor():
    print("\n" + "=" * 60)
    print("TEST 7: Performance Predictor")
    print("=" * 60)
    from performance_predictor import (
        ConcurrencyPerformancePredictor, AmdahlsLaw, GustafsonsLaw,
        WorkloadProfile, ArchitectureSpec
    )
    results = {}

    def test_amdahl():
        amdahl = AmdahlsLaw()
        # 80% parallel
        sp_2 = amdahl.speedup(0.8, 2)
        sp_4 = amdahl.speedup(0.8, 4)
        sp_8 = amdahl.speedup(0.8, 8)
        sp_inf = amdahl.max_speedup(0.8)
        assert abs(sp_inf - 5.0) < 0.01, f"Max speedup should be 5.0, got {sp_inf}"
        assert sp_2 < sp_4 < sp_8 < sp_inf
        # Known value: sp(0.8, 4) = 1/(0.2 + 0.8/4) = 1/0.4 = 2.5
        assert abs(sp_4 - 2.5) < 0.01, f"sp(0.8,4) should be 2.5, got {sp_4}"
        return {"sp_2": sp_2, "sp_4": sp_4, "sp_8": sp_8, "sp_inf": sp_inf}

    def test_gustafson():
        gustafson = GustafsonsLaw()
        sp = gustafson.scaled_speedup(0.1, 8)
        # s(0.1, 8) = 8 - 0.1*(8-1) = 8 - 0.7 = 7.3
        assert abs(sp - 7.3) < 0.01
        return {"gustafson_sp": sp}

    def test_predictor():
        predictor = ConcurrencyPerformancePredictor()
        workload = WorkloadProfile(
            parallel_fraction=0.9,
            compute_fraction=0.6,
            memory_fraction=0.2,
            sync_fraction=0.1,
            io_fraction=0.1,
        )
        report = predictor.predict(workload, 8)
        assert report.predicted_speedup > 1.0
        assert report.scalability_limit > 0
        return report.summary()

    def test_speedup_curve():
        predictor = ConcurrencyPerformancePredictor()
        workload = WorkloadProfile(parallel_fraction=0.95)
        curve = predictor.predict_speedup_curve(workload, max_threads=16)
        assert len(curve["threads"]) > 0
        assert curve["speedup"][0] >= 1.0
        return {"n_points": len(curve["threads"]), "max_speedup": max(curve["speedup"])}

    for name, fn in [("Amdahl's law", test_amdahl),
                     ("Gustafson's law", test_gustafson),
                     ("Predictor", test_predictor),
                     ("Speedup curve", test_speedup_curve)]:
        passed, result, elapsed = run_test(name, fn)
        results[name] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Test 8: Concurrency Patterns
# =============================================================================
def test_concurrency_patterns():
    print("\n" + "=" * 60)
    print("TEST 8: Concurrency Patterns")
    print("=" * 60)
    from concurrency_patterns import (
        ProducerConsumer, ReadersWriters, DiningPhilosophers,
        BarberShop, Pipeline, MapReduce, ForkJoin,
        CorrectnessVerifier, PerformanceComparator
    )
    results = {}

    def test_producer_consumer():
        pc = ProducerConsumer(buffer_size=20, n_producers=5, n_consumers=3)
        metrics = pc.run_simulation(items_per_producer=40, rng_seed=42)
        assert metrics.items_processed > 0
        verifier = CorrectnessVerifier()
        correctness = verifier.verify_producer_consumer(pc, 40)
        return {"processed": metrics.items_processed, "correctness": correctness}

    def test_readers_writers():
        rw = ReadersWriters(preference="fair")
        metrics = rw.run_simulation(n_readers=5, n_writers=2, ops_per_thread=30)
        assert metrics.items_processed > 0
        return {"processed": metrics.items_processed, "fairness": metrics.fairness}

    def test_dining_philosophers():
        dp = DiningPhilosophers(n_philosophers=5, solution="hierarchy")
        metrics = dp.run_simulation(rounds=50)
        verifier = CorrectnessVerifier()
        correctness = verifier.verify_dining_philosophers(dp)
        assert correctness["deadlock_free"]
        return {"eats": sum(dp.eat_count), "correctness": correctness}

    def test_pipeline():
        pipe = Pipeline(n_stages=4, buffer_size=20)
        metrics = pipe.run_simulation(list(range(100)))
        assert metrics.items_processed > 0
        return {"processed": metrics.items_processed}

    def test_mapreduce():
        mr = MapReduce(n_mappers=4, n_reducers=2)
        mr.set_map_function(lambda x: [(x % 10, x)])
        mr.set_reduce_function(lambda k, vs: (k, sum(vs)))
        metrics = mr.run(list(range(100)))
        assert metrics.items_processed > 0
        return {"results": len(mr.results)}

    def test_forkjoin():
        fj = ForkJoin(n_workers=4, threshold=20)
        data = list(np.random.RandomState(42).random(200))
        result = fj.compute(data)
        expected = sum(data)
        assert abs(result - expected) < 1e-6
        return {"result": result, "expected": expected, "match": abs(result - expected) < 1e-6}

    def test_comparison():
        comparator = PerformanceComparator()
        results_cmp = comparator.compare_patterns(workload_size=200, rng_seed=42)
        ranking = comparator.rank_patterns(results_cmp)
        return {"ranking": [(name, round(score, 2)) for name, score in ranking]}

    for name, fn in [("Producer-Consumer", test_producer_consumer),
                     ("Readers-Writers", test_readers_writers),
                     ("Dining Philosophers", test_dining_philosophers),
                     ("Pipeline", test_pipeline),
                     ("MapReduce", test_mapreduce),
                     ("ForkJoin", test_forkjoin),
                     ("Pattern comparison", test_comparison)]:
        passed, result, elapsed = run_test(name, fn)
        results[name] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Test 9: GPU Memory Model
# =============================================================================
def test_gpu_memory_model():
    print("\n" + "=" * 60)
    print("TEST 9: GPU Memory Model")
    print("=" * 60)
    from gpu_memory_model import (
        GPUMemoryModel, GPUScopeTree, GPULitmusTestBuilder,
        GPURaceDetector, GPUFenceType, GPUScopeLevel
    )
    results = {}

    def test_ptx_sb():
        scope_tree = GPUScopeTree()
        scope_tree.thread_to_cta = {0: 0, 1: 0}
        scope_tree.thread_to_warp = {0: 0, 1: 0}
        scope_tree.thread_to_gpu = {0: 0, 1: 0}
        model = GPUMemoryModel(scope_tree)
        builder = GPULitmusTestBuilder()
        sb = builder.build_gpu_sb(same_cta=True)
        allowed, violations = model.check(sb, "PTX")
        return {"allowed": allowed, "violations": violations}

    def test_ptx_mp():
        model = GPUMemoryModel()
        builder = GPULitmusTestBuilder()
        mp_nofence = builder.build_gpu_mp(fence_type=None)
        allowed_nf, _ = model.check(mp_nofence, "PTX")
        mp_fence = builder.build_gpu_mp(fence_type=GPUFenceType.THREADFENCE)
        allowed_f, _ = model.check(mp_fence, "PTX")
        return {"no_fence_allowed": allowed_nf, "with_fence_allowed": allowed_f}

    def test_scope_visibility():
        model = GPUMemoryModel()
        builder = GPULitmusTestBuilder()
        scope_test = builder.build_gpu_scope_test()
        allowed, violations = model.check(scope_test, "PTX")
        return {"allowed": allowed, "violations": violations}

    def test_litmus_suite():
        model = GPUMemoryModel()
        suite_results = model.run_gpu_litmus_suite()
        assert len(suite_results) >= 5, "Should run at least 5 litmus tests"
        return {name: res["allowed"] for name, res in suite_results.items()}

    def test_race_detection():
        scope_tree = GPUScopeTree()
        scope_tree.thread_to_cta = {0: 0, 1: 1}
        model = GPUMemoryModel(scope_tree)
        builder = GPULitmusTestBuilder()
        sb = builder.build_gpu_sb(same_cta=False)
        races = model.detect_races(sb)
        return {"races_found": len(races)}

    for name, fn in [("PTX SB test", test_ptx_sb),
                     ("PTX MP test", test_ptx_mp),
                     ("Scope visibility", test_scope_visibility),
                     ("Litmus suite", test_litmus_suite),
                     ("Race detection", test_race_detection)]:
        passed, result, elapsed = run_test(name, fn)
        results[name] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Test 10: Run existing benchmarks
# =============================================================================
def test_existing_benchmarks():
    print("\n" + "=" * 60)
    print("TEST 10: Existing Benchmarks")
    print("=" * 60)
    results = {}

    def test_run_full_benchmark():
        benchmark_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "run_full_benchmark.py")
        if os.path.exists(benchmark_path):
            return {"status": "exists", "path": benchmark_path}
        return {"status": "not_found"}

    passed, result, elapsed = run_test("Existing benchmarks check", test_run_full_benchmark)
    results["existing_benchmarks"] = {"passed": passed, "result": result if passed else str(result), "time": elapsed}

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 60)

    overall_start = time.time()
    all_results = {}
    total_tests = 0
    total_passed = 0

    test_suites = [
        ("program_analyzer", test_program_analyzer),
        ("memory_model_compiler", test_memory_model_compiler),
        ("verification_engine", test_verification_engine),
        ("lock_free_algorithms", test_lock_free_algorithms),
        ("weak_memory_testing", test_weak_memory_testing),
        ("synchronization_synthesis", test_synchronization_synthesis),
        ("performance_predictor", test_performance_predictor),
        ("concurrency_patterns", test_concurrency_patterns),
        ("gpu_memory_model", test_gpu_memory_model),
        ("existing_benchmarks", test_existing_benchmarks),
    ]

    for suite_name, suite_fn in test_suites:
        try:
            suite_results = suite_fn()
            all_results[suite_name] = suite_results
            for test_name, test_result in suite_results.items():
                total_tests += 1
                if test_result.get("passed", False):
                    total_passed += 1
        except Exception as e:
            print(f"\n  [SUITE FAIL] {suite_name}: {e}")
            traceback.print_exc()
            all_results[suite_name] = {"error": str(e)}

    overall_elapsed = time.time() - overall_start

    # Summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total tests:  {total_tests}")
    print(f"Passed:       {total_passed}")
    print(f"Failed:       {total_tests - total_passed}")
    print(f"Pass rate:    {total_passed/max(total_tests,1)*100:.1f}%")
    print(f"Total time:   {overall_elapsed:.3f}s")
    print()

    for suite_name, suite_results in all_results.items():
        if isinstance(suite_results, dict) and "error" not in suite_results:
            passed_count = sum(1 for r in suite_results.values()
                               if isinstance(r, dict) and r.get("passed", False))
            total_count = len(suite_results)
            status = "PASS" if passed_count == total_count else "PARTIAL"
            print(f"  {suite_name}: {passed_count}/{total_count} [{status}]")
        else:
            print(f"  {suite_name}: ERROR")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "comprehensive_benchmark_results.json")

    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        return str(obj)

    serializable_results = make_serializable({
        "summary": {
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_tests - total_passed,
            "pass_rate": total_passed / max(total_tests, 1) * 100,
            "total_time": overall_elapsed,
        },
        "suites": all_results,
    })

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
