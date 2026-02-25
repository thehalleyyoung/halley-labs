#!/usr/bin/env python3
"""
LITMUS∞ Phase B2 Comprehensive Experiment Runner.

Runs all expanded experiments:
1. Full Z3 SMT-LIB certificate extraction (750/750 with proofs)
2. Compositional reasoning analysis (multi-pattern programs)
3. Expanded benchmark (500+ snippets with stratified accuracy)
4. Severity classification with root cause analysis
5. All prior validations (DSL-.cat, herd7, fence proofs)

All results saved to paper_results_v6/ as JSON.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def run_all_phase_b2_experiments():
    """Run complete Phase B2 experiment suite."""
    print("=" * 70)
    print("LITMUS∞ Phase B2 — Comprehensive Experiment Suite")
    print("=" * 70)

    os.makedirs('paper_results_v6', exist_ok=True)
    all_results = {}
    t0 = time.time()

    # ── 1. Full Z3 SMT-LIB Certificate Extraction ──
    print("\n[1/7] Full Z3 SMT-LIB Certificate Extraction (750 pairs)...")
    from smtlib_certificate_extractor import generate_all_smtlib_certificates
    t1 = time.time()
    cert_report = generate_all_smtlib_certificates('paper_results_v6/smtlib_certificates')
    t1e = time.time() - t1
    print(f"  Certified: {cert_report['certified']}/{cert_report['total_pairs']} "
          f"({cert_report['certificate_coverage_pct']}%)")
    print(f"  SAT witnesses: {cert_report['sat_witnesses']}")
    print(f"  UNSAT proofs: {cert_report['unsat_proofs']}")
    print(f"  Avg UNSAT core size: {cert_report['avg_unsat_core_size']}")
    print(f"  Completed in {t1e:.1f}s")
    all_results['smtlib_certificates'] = {
        'total_pairs': cert_report['total_pairs'],
        'certified': cert_report['certified'],
        'coverage_pct': cert_report['certificate_coverage_pct'],
        'sat_witnesses': cert_report['sat_witnesses'],
        'unsat_proofs': cert_report['unsat_proofs'],
        'avg_unsat_core_size': cert_report['avg_unsat_core_size'],
        'wilson_95ci': cert_report['wilson_95ci'],
        'time_s': round(t1e, 1),
    }

    # ── 2. Compositional Reasoning Analysis ──
    print("\n[2/7] Compositional Reasoning Analysis...")
    from compositional_reasoning import run_compositional_analysis
    t2 = time.time()
    comp_report = run_compositional_analysis('paper_results_v6')
    t2e = time.time() - t2
    stats = comp_report.get('statistics', {})
    print(f"  Disjoint compositions: {stats.get('disjoint_compositions', 0)}")
    print(f"  Shared-variable interactions: {stats.get('shared_variable_interactions', 0)}")
    print(f"  Safe: {stats.get('safe_results', 0)}, Unsafe: {stats.get('unsafe_results', 0)}")
    print(f"  Completed in {t2e:.1f}s")
    all_results['compositional_analysis'] = {
        'n_examples': comp_report.get('n_examples', 0),
        'total_analyses': comp_report.get('total_analyses', 0),
        'statistics': stats,
        'time_s': round(t2e, 1),
    }

    # ── 3. Expanded Benchmark (500+ snippets) ──
    print("\n[3/7] Expanded Benchmark Suite (500+ snippets)...")
    from expanded_benchmark_v2 import EXPANDED_BENCHMARK_SNIPPETS, run_expanded_benchmark
    from ast_analyzer import ast_analyze_code
    t3 = time.time()
    bench_results, bench_summary = run_expanded_benchmark(ast_analyze_code)
    t3e = time.time() - t3
    print(f"  Total snippets: {bench_summary['total']}")
    print(f"  Exact match: {bench_summary['exact_correct']}/{bench_summary['total']} "
          f"({bench_summary['exact_accuracy']}%)")
    print(f"  Top-3 match: {bench_summary['top3_correct']}/{bench_summary['total']} "
          f"({bench_summary['top3_accuracy']}%)")
    print(f"  Exact Wilson 95% CI: {bench_summary['exact_wilson_95ci']}")
    print(f"  Top-3 Wilson 95% CI: {bench_summary['top3_wilson_95ci']}")
    print(f"  Failures: {bench_summary['failure_count']}")
    print(f"  Completed in {t3e:.1f}s")

    # Per-pattern breakdown
    print(f"\n  Per-pattern accuracy:")
    for pat, stats in sorted(bench_summary['per_pattern'].items()):
        print(f"    {pat:20s}: {stats['exact']}/{stats['total']} "
              f"({stats['exact_accuracy']}%) CI {stats['wilson_95ci']}")

    # Per-category breakdown
    print(f"\n  Per-category accuracy:")
    for cat, stats in sorted(bench_summary['per_category'].items()):
        print(f"    {cat:20s}: {stats['exact']}/{stats['total']} "
              f"({stats['exact_accuracy']}%) CI {stats['wilson_95ci']}")

    with open('paper_results_v6/expanded_benchmark.json', 'w') as f:
        json.dump({'results': bench_results, 'summary': bench_summary},
                  f, indent=2, default=str)
    all_results['expanded_benchmark'] = {
        'total': bench_summary['total'],
        'exact_correct': bench_summary['exact_correct'],
        'top3_correct': bench_summary['top3_correct'],
        'exact_accuracy': bench_summary['exact_accuracy'],
        'top3_accuracy': bench_summary['top3_accuracy'],
        'exact_wilson_95ci': bench_summary['exact_wilson_95ci'],
        'top3_wilson_95ci': bench_summary['top3_wilson_95ci'],
        'failure_count': bench_summary['failure_count'],
        'per_pattern': bench_summary['per_pattern'],
        'per_category': bench_summary['per_category'],
        'time_s': round(t3e, 1),
    }

    # ── 4. Severity Classification ──
    print("\n[4/7] Severity Classification of Unsafe Pairs...")
    from severity_classification import classify_all_unsafe_pairs
    t4 = time.time()
    severity_report = classify_all_unsafe_pairs()
    t4e = time.time() - t4
    print(f"  Total unsafe pairs: {severity_report['total_unsafe_pairs']}")
    for sev, count in sorted(severity_report['severity_counts'].items()):
        pct = 100 * count / max(severity_report['total_unsafe_pairs'], 1)
        print(f"    {sev}: {count} ({pct:.1f}%)")
    print(f"  Completed in {t4e:.1f}s")

    with open('paper_results_v6/severity_classification.json', 'w') as f:
        json.dump(severity_report, f, indent=2, default=str)
    all_results['severity_classification'] = {
        'total_unsafe': severity_report['total_unsafe_pairs'],
        'severity_counts': severity_report['severity_counts'],
        'z3_certified': severity_report.get('z3_certified', 0),
        'time_s': round(t4e, 1),
    }

    # ── 5. DSL-.cat Correspondence ──
    print("\n[5/7] DSL-to-.cat Formal Correspondence...")
    from dsl_cat_correspondence import validate_all_models
    t5 = time.time()
    dsl_report = validate_all_models()
    t5e = time.time() - t5
    print(f"  Agreement: {dsl_report['total_agree']}/{dsl_report['total_checks']} "
          f"({dsl_report['overall_agreement_rate']}%)")
    print(f"  Wilson 95% CI: {dsl_report['overall_wilson_95ci']}")
    print(f"  Completed in {t5e:.1f}s")

    with open('paper_results_v6/dsl_cat_correspondence.json', 'w') as f:
        json.dump(dsl_report, f, indent=2, default=str)
    all_results['dsl_cat_correspondence'] = {
        'agree': dsl_report['total_agree'],
        'total': dsl_report['total_checks'],
        'rate': dsl_report['overall_agreement_rate'],
        'wilson_95ci': dsl_report['overall_wilson_95ci'],
        'time_s': round(t5e, 1),
    }

    # ── 6. herd7 Cross-Validation ──
    print("\n[6/7] herd7 Cross-Validation...")
    from herd7_validation import validate_against_herd7
    t6 = time.time()
    herd7_report = validate_against_herd7()
    t6e = time.time() - t6
    print(f"  Agreement: {herd7_report['agreements']}/{herd7_report['total_checks']} "
          f"({herd7_report['agreement_rate']*100:.1f}%)")
    print(f"  Completed in {t6e:.1f}s")

    with open('paper_results_v6/herd7_validation.json', 'w') as f:
        json.dump(herd7_report, f, indent=2, default=str)
    all_results['herd7_validation'] = {
        'agree': herd7_report['agreements'],
        'total': herd7_report['total_checks'],
        'rate': round(herd7_report['agreement_rate'] * 100, 1),
        'time_s': round(t6e, 1),
    }

    # ── 7. Portability Matrix ──
    print("\n[7/7] Full Portability Matrix...")
    from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest, recommend_fence
    t7 = time.time()
    matrix = []
    safe_count = 0
    unsafe_count = 0
    all_archs = ['x86', 'sparc', 'arm', 'riscv',
                 'opencl_wg', 'opencl_dev', 'vulkan_wg', 'vulkan_dev',
                 'ptx_cta', 'ptx_gpu']

    for pat_name in sorted(PATTERNS.keys()):
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )
        for arch_name in all_archs:
            allowed, _ = verify_test(lt, ARCHITECTURES[arch_name])
            safe = not allowed
            if safe:
                safe_count += 1
            else:
                unsafe_count += 1
            matrix.append({
                'pattern': pat_name,
                'arch': arch_name,
                'safe': safe,
            })

    t7e = time.time() - t7
    print(f"  Total: {len(matrix)} pairs ({safe_count} safe, {unsafe_count} unsafe)")
    print(f"  Completed in {t7e:.1f}s")

    with open('paper_results_v6/portability_matrix.json', 'w') as f:
        json.dump({
            'total': len(matrix),
            'safe': safe_count,
            'unsafe': unsafe_count,
            'matrix': matrix,
        }, f, indent=2)
    all_results['portability_matrix'] = {
        'total': len(matrix),
        'safe': safe_count,
        'unsafe': unsafe_count,
        'time_s': round(t7e, 1),
    }

    # ── Summary ──
    total_time = time.time() - t0
    all_results['total_time_s'] = round(total_time, 1)

    with open('paper_results_v6/experiment_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE B2 EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"SMT-LIB Certificates: {all_results['smtlib_certificates']['certified']}/"
          f"{all_results['smtlib_certificates']['total_pairs']} "
          f"({all_results['smtlib_certificates']['coverage_pct']}%)")
    print(f"Compositional Analysis: {all_results['compositional_analysis']['total_analyses']} analyses")
    print(f"Expanded Benchmark: {all_results['expanded_benchmark']['exact_correct']}/"
          f"{all_results['expanded_benchmark']['total']} exact "
          f"({all_results['expanded_benchmark']['exact_accuracy']}%)")
    print(f"Severity Classification: {all_results['severity_classification']['total_unsafe']} pairs")
    print(f"DSL-.cat: {all_results['dsl_cat_correspondence']['agree']}/"
          f"{all_results['dsl_cat_correspondence']['total']} "
          f"({all_results['dsl_cat_correspondence']['rate']}%)")
    print(f"herd7: {all_results['herd7_validation']['agree']}/"
          f"{all_results['herd7_validation']['total']} "
          f"({all_results['herd7_validation']['rate']}%)")
    print(f"Portability Matrix: {all_results['portability_matrix']['safe']} safe, "
          f"{all_results['portability_matrix']['unsafe']} unsafe")
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"All results saved to paper_results_v6/")

    return all_results


if __name__ == '__main__':
    run_all_phase_b2_experiments()
