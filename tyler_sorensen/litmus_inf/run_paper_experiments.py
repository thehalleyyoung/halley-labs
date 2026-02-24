#!/usr/bin/env python3
"""Generate all experimental results for LITMUS∞ paper (v4)."""

import json
import os
import sys
import time
import statistics

sys.path.insert(0, os.path.dirname(__file__))

os.makedirs('paper_results_v4', exist_ok=True)

def run_portability_analysis():
    """Full 570-pair portability analysis."""
    from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest, recommend_fence

    results = []
    safe_count = 0
    fail_count = 0

    for pat_name in sorted(PATTERNS.keys()):
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )
        for arch_name in sorted(ARCHITECTURES.keys()):
            arch_model = ARCHITECTURES[arch_name]
            allowed, _ = verify_test(lt, arch_model)
            safe = not allowed
            fence = None
            if not safe:
                fence = recommend_fence(lt, arch_name, arch_model)

            if safe:
                safe_count += 1
            else:
                fail_count += 1

            results.append({
                'pattern': pat_name,
                'arch': arch_name,
                'safe': safe,
                'fence': fence,
            })

    data = {
        'total': len(results),
        'safe': safe_count,
        'fail': fail_count,
        'results': results,
    }

    with open('paper_results_v4/portability_matrix.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Portability: {len(results)} pairs, {safe_count} safe, {fail_count} unsafe")
    return data


def run_gpu_scope_analysis():
    """GPU scope mismatch detection."""
    from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest

    cpu_models = ['x86', 'sparc', 'arm', 'riscv']
    gpu_models = [k for k in ARCHITECTURES if k not in cpu_models]

    critical = []
    warnings = []

    for pat_name in sorted(PATTERNS.keys()):
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )

        # Check all CPUs safe
        all_cpu_safe = all(
            not verify_test(lt, ARCHITECTURES[m])[0]
            for m in cpu_models
        )
        if not all_cpu_safe:
            continue

        # Check GPU failures
        gpu_results = {}
        for gm in gpu_models:
            allowed, _ = verify_test(lt, ARCHITECTURES[gm])
            gpu_results[gm] = not allowed  # safe?

        all_gpu_fail = all(not v for v in gpu_results.values())
        some_gpu_fail = any(not v for v in gpu_results.values())

        if all_gpu_fail:
            critical.append({
                'pattern': pat_name,
                'severity': 'critical',
                'mode': 'all GPU fail',
                'gpu_results': gpu_results,
            })
        elif some_gpu_fail:
            failing = [k for k, v in gpu_results.items() if not v]
            warnings.append({
                'pattern': pat_name,
                'severity': 'warning',
                'mode': 'partial GPU fail',
                'failing_models': failing,
                'gpu_results': gpu_results,
            })

    data = {
        'critical': critical,
        'warnings': warnings,
        'n_critical': len(critical),
        'n_warnings': len(warnings),
    }

    with open('paper_results_v4/gpu_scope_mismatches.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  GPU scope: {len(critical)} critical, {len(warnings)} warnings")
    return data


def run_timing():
    """Timing benchmark (20 runs)."""
    from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest

    n_runs = 20
    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        for pat_name, pat_def in PATTERNS.items():
            n_threads = max(op.thread for op in pat_def['ops']) + 1
            lt = LitmusTest(
                name=pat_name, n_threads=n_threads,
                addresses=pat_def['addresses'], ops=pat_def['ops'],
                forbidden=pat_def['forbidden'],
            )
            for arch_model in ARCHITECTURES.values():
                verify_test(lt, arch_model)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    data = {
        'mean_ms': statistics.mean(times),
        'stdev_ms': statistics.stdev(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'n_runs': n_runs,
        'all_times_ms': times,
    }

    with open('paper_results_v4/timing.json', 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Timing: {data['mean_ms']:.1f}ms mean (stdev {data['stdev_ms']:.1f}ms)")
    return data


def run_benchmark_suite():
    """AST analyzer benchmark on 96 snippets."""
    from benchmark_suite import run_benchmark, BENCHMARK_SNIPPETS
    from ast_analyzer import ASTAnalyzer

    analyzer = ASTAnalyzer()

    def analyzer_func(code):
        result = analyzer.analyze(code)
        return [(m.pattern_name, m.confidence) for m in result.patterns_found]

    results_list, summary = run_benchmark(analyzer.analyze)

    # Save to v4
    with open('paper_results_v4/ast_benchmark_results.json', 'w') as f:
        json.dump({'results': results_list, 'summary': summary}, f, indent=2, default=str)

    exact = summary['exact_correct']
    top3 = summary['top3_correct']
    total = summary['total']

    print(f"  AST benchmark: {exact}/{total} exact ({100*exact/total:.1f}%), {top3}/{total} top-3 ({100*top3/total:.1f}%)")
    return {'results': results_list, 'summary': summary}


def run_fence_analysis():
    """Per-thread fence cost analysis."""
    from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest, recommend_fence

    fence_results = []
    arch_costs = {
        'arm': {'dmb ishst': 1, 'dmb ishld': 2, 'dmb ish': 4},
        'riscv': {'fence r,r': 1, 'fence w,w': 1, 'fence r,w': 1, 'fence w,r': 2, 'fence rw,rw': 4},
    }

    for pat_name in sorted(PATTERNS.keys()):
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )

        for arch in ['arm', 'riscv']:
            arch_model = ARCHITECTURES[arch]
            allowed, _ = verify_test(lt, arch_model)
            if not allowed:
                continue  # safe, no fence needed

            fence = recommend_fence(lt, arch, arch_model)
            if not fence:
                continue

            # Parse fence recommendation
            full_cost = arch_costs[arch].get('dmb ish' if arch == 'arm' else 'fence rw,rw', 4)
            min_cost = 0
            for thread_fence in fence.split(';'):
                tf = thread_fence.strip().lower()
                for fname, fcost in arch_costs[arch].items():
                    if fname in tf:
                        min_cost += fcost
                        break
                else:
                    min_cost += full_cost

            max_possible = full_cost * n_threads
            savings = (max_possible - min_cost) / max_possible * 100 if max_possible > 0 else 0

            fence_results.append({
                'pattern': pat_name,
                'arch': arch,
                'fence': fence,
                'min_cost': min_cost,
                'full_cost': max_possible,
                'savings_pct': round(savings, 1),
            })

    with open('paper_results_v4/fence_optimization.json', 'w') as f:
        json.dump(fence_results, f, indent=2)

    arm_savings = [r['savings_pct'] for r in fence_results if r['arch'] == 'arm']
    rv_savings = [r['savings_pct'] for r in fence_results if r['arch'] == 'riscv']

    print(f"  Fences: {len(fence_results)} pairs, "
          f"ARM avg {statistics.mean(arm_savings):.1f}%, "
          f"RISC-V avg {statistics.mean(rv_savings):.1f}%")
    return fence_results


def run_custom_models():
    """Custom model DSL analysis."""
    from model_dsl import register_model, check_custom, get_registry
    from portcheck import PATTERNS

    models_dsl = {
        'POWER': """model POWER {
    description "IBM POWER memory model"
    relaxes W->R, W->W, R->R, R->W
    preserves deps
    not multi-copy-atomic
    fence hwsync (cost=8) { orders W->R, W->W, R->R, R->W }
    fence lwsync (cost=4) { orders W->W, R->R, R->W }
    fence isync (cost=2) { orders R->R }
}""",
        'Alpha': """model Alpha {
    description "DEC Alpha memory model"
    relaxes W->R, W->W, R->R, R->W
    not multi-copy-atomic
    fence mb (cost=8) { orders W->R, W->W, R->R, R->W }
    fence wmb (cost=4) { orders W->W }
    fence rmb (cost=4) { orders R->R }
}""",
        'C11_Relaxed': """model C11_Relaxed {
    description "C11 with relaxed atomics"
    relaxes W->R, W->W, R->R, R->W
    preserves deps
    not multi-copy-atomic
    fence seq_cst (cost=8) { orders W->R, W->W, R->R, R->W }
    fence release (cost=4) { orders W->W, R->W }
    fence acquire (cost=4) { orders R->R, R->W }
    fence acq_rel (cost=6) { orders W->W, R->R, R->W }
}""",
        'SC': """model SC {
    description "Sequential consistency"
    fence full (cost=8) { orders W->R, W->W, R->R, R->W }
}""",
    }

    for name, dsl in models_dsl.items():
        register_model(dsl)

    model_results = {}
    for model_name in ['SC', 'POWER', 'Alpha', 'C11_Relaxed']:
        safe = 0
        unsafe = 0
        for pat_name in sorted(PATTERNS.keys()):
            if pat_name.startswith('gpu_'):
                continue
            result = check_custom(pat_name, model_name)
            if result.get('safe', True):
                safe += 1
            else:
                unsafe += 1

        model_results[model_name] = {'safe': safe, 'unsafe': unsafe}

    # Model comparisons
    registry = get_registry()
    comparisons = {}
    pairs = [('POWER', 'ARM'), ('Alpha', 'ARM'), ('C11_Relaxed', 'ARM')]
    for m1, m2 in pairs:
        try:
            diff = registry.compare_models(m1, m2)
            comparisons[f'{m1}_vs_{m2}'] = diff
        except Exception:
            comparisons[f'{m1}_vs_{m2}'] = {'error': 'comparison failed'}

    data = {
        'model_results': model_results,
        'comparisons': comparisons,
    }

    with open('paper_results_v4/custom_model_results.json', 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"  Custom models: {json.dumps(model_results)}")
    return data


def run_model_boundaries():
    """Model boundary analysis."""
    from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest

    boundaries = [
        ('x86', 'sparc', 'TSO→PSO'),
        ('sparc', 'arm', 'PSO→ARM'),
        ('arm', 'riscv', 'ARM→RISC-V'),
    ]

    boundary_results = {}
    for stronger, weaker, label in boundaries:
        discriminators = []
        for pat_name in sorted(PATTERNS.keys()):
            if pat_name.startswith('gpu_'):
                continue
            pat_def = PATTERNS[pat_name]
            n_threads = max(op.thread for op in pat_def['ops']) + 1
            lt = LitmusTest(
                name=pat_name, n_threads=n_threads,
                addresses=pat_def['addresses'], ops=pat_def['ops'],
                forbidden=pat_def['forbidden'],
            )

            s_allowed, _ = verify_test(lt, ARCHITECTURES[stronger])
            w_allowed, _ = verify_test(lt, ARCHITECTURES[weaker])

            if not s_allowed and w_allowed:
                discriminators.append(pat_name)

        boundary_results[label] = {
            'count': len(discriminators),
            'patterns': discriminators,
        }

    # GPU boundary
    gpu_disc = []
    for pat_name in sorted(PATTERNS.keys()):
        if not pat_name.startswith('gpu_'):
            continue
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )

        dev_allowed, _ = verify_test(lt, ARCHITECTURES.get('opencl_dev', ARCHITECTURES.get('vulkan_dev')))
        wg_allowed, _ = verify_test(lt, ARCHITECTURES.get('opencl_wg', ARCHITECTURES.get('vulkan_wg')))

        if not dev_allowed and wg_allowed:
            gpu_disc.append(pat_name)

    boundary_results['GPU-Dev→GPU-WG'] = {
        'count': len(gpu_disc),
        'patterns': gpu_disc,
    }

    with open('paper_results_v4/model_boundaries.json', 'w') as f:
        json.dump(boundary_results, f, indent=2)

    print(f"  Boundaries: " + ", ".join(f"{k}: {v['count']}" for k, v in boundary_results.items()))
    return boundary_results


if __name__ == '__main__':
    print("=" * 70)
    print("LITMUS∞ Paper Experiments (v5)")
    print("=" * 70)

    print("\n[1/9] Full portability analysis...")
    run_portability_analysis()

    print("\n[2/9] GPU scope mismatch detection...")
    run_gpu_scope_analysis()

    print("\n[3/9] Timing benchmark...")
    run_timing()

    print("\n[4/9] AST benchmark suite...")
    run_benchmark_suite()

    print("\n[5/9] Fence cost analysis...")
    run_fence_analysis()

    print("\n[6/9] Custom models & boundaries...")
    run_custom_models()
    run_model_boundaries()

    print("\n[7/9] Mismatch analysis...")
    from mismatch_analysis import run_full_analysis as run_mismatch
    run_mismatch()

    print("\n[8/9] SMT validation...")
    from smt_validation import run_full_smt_validation
    run_full_smt_validation()

    print("\n[9/9] Completeness analysis...")
    from completeness_analysis import run_completeness_analysis
    run_completeness_analysis()

    print("\n" + "=" * 70)
    print("All results saved to paper_results_v4/")
    print("=" * 70)
