#!/usr/bin/env python3
"""
LITMUS∞ Phase B Comprehensive Experiment Runner.

Runs all experiments for the paper revision:
1. Full Z3 certificate coverage (750/750 target)
2. Severity classification of all 342 unsafe pairs
3. DSL-.cat formal correspondence for TSO, ARM, RISC-V
4. herd7 cross-validation
5. Fence sufficiency proofs
6. Model discriminator synthesis

All results saved to paper_results_v5/ as JSON.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def run_all_phase_b_experiments():
    """Run all Phase B experiments and save comprehensive results."""
    print("=" * 70)
    print("LITMUS∞ Phase B — Full Experiment Suite")
    print("=" * 70)

    os.makedirs('paper_results_v5', exist_ok=True)
    all_results = {}
    t0 = time.time()

    # ── 1. Full Z3 Certificate Coverage ──
    print("\n[1/6] Full Z3 Certificate Coverage (all pattern × model pairs)...")
    from smt_validation import cross_validate_all_750_smt, Z3_AVAILABLE
    if not Z3_AVAILABLE:
        print("  ERROR: Z3 not available")
        return
    t1 = time.time()
    cert_report = cross_validate_all_750_smt()
    t1e = time.time() - t1
    print(f"  Certified: {cert_report['certified']}/{cert_report['total_pairs']} "
          f"({cert_report['certificate_coverage_pct']}%)")
    print(f"  UNSAT (safe): {cert_report['cert_safe_unsat']}, "
          f"SAT (unsafe): {cert_report['cert_unsafe_sat']}")
    print(f"  Agreement with enumeration: {cert_report['agree']}/{cert_report['agree']+cert_report['disagree']} "
          f"({cert_report['agreement_rate']}%)")
    print(f"  Wilson 95% CI: {cert_report['wilson_95ci']}")
    print(f"  Timeouts: {cert_report['timeouts']}")
    print(f"  Completed in {t1e:.1f}s")

    with open('paper_results_v5/universal_certificates.json', 'w') as f:
        json.dump(cert_report, f, indent=2, default=str)
    all_results['universal_certificates'] = {
        'total_pairs': cert_report['total_pairs'],
        'certified': cert_report['certified'],
        'coverage_pct': cert_report['certificate_coverage_pct'],
        'safe': cert_report['cert_safe_unsat'],
        'unsafe': cert_report['cert_unsafe_sat'],
        'agreement_rate': cert_report['agreement_rate'],
        'wilson_95ci': cert_report['wilson_95ci'],
        'time_s': round(t1e, 1),
    }

    # ── 2. Severity Classification ──
    print("\n[2/6] Severity Classification of Unsafe Pairs...")
    from severity_classification import classify_all_unsafe_pairs
    t2 = time.time()
    severity_report = classify_all_unsafe_pairs()
    t2e = time.time() - t2
    print(f"  Total unsafe pairs: {severity_report['total_unsafe_pairs']}")
    for sev, count in sorted(severity_report['severity_counts'].items()):
        pct = 100 * count / max(severity_report['total_unsafe_pairs'], 1)
        print(f"  {sev}: {count} ({pct:.1f}%)")
    print(f"  Z3 certified: {severity_report['z3_certified']}")
    print(f"  Completed in {t2e:.1f}s")

    with open('paper_results_v5/severity_classification.json', 'w') as f:
        json.dump(severity_report, f, indent=2, default=str)
    all_results['severity_classification'] = {
        'total_unsafe': severity_report['total_unsafe_pairs'],
        'severity_counts': severity_report['severity_counts'],
        'z3_certified': severity_report['z3_certified'],
        'time_s': round(t2e, 1),
    }

    # ── 3. DSL-.cat Formal Correspondence ──
    print("\n[3/6] DSL-to-.cat Formal Correspondence Validation...")
    from dsl_cat_correspondence import validate_all_models
    t3 = time.time()
    dsl_cat_report = validate_all_models()
    t3e = time.time() - t3
    print(f"  Total checks: {dsl_cat_report['total_checks']}")
    print(f"  Agreement: {dsl_cat_report['total_agree']}/{dsl_cat_report['total_checks']} "
          f"({dsl_cat_report['overall_agreement_rate']}%)")
    print(f"  Wilson 95% CI: {dsl_cat_report['overall_wilson_95ci']}")
    for model_name, model_data in dsl_cat_report['per_model'].items():
        print(f"  {model_name}: {model_data['agree']}/{model_data['total']} "
              f"({model_data['agreement_rate']}%)")
    print(f"  Completed in {t3e:.1f}s")

    with open('paper_results_v5/dsl_cat_correspondence.json', 'w') as f:
        json.dump(dsl_cat_report, f, indent=2, default=str)
    all_results['dsl_cat_correspondence'] = {
        'total_checks': dsl_cat_report['total_checks'],
        'agree': dsl_cat_report['total_agree'],
        'agreement_rate': dsl_cat_report['overall_agreement_rate'],
        'wilson_95ci': dsl_cat_report['overall_wilson_95ci'],
        'per_model': {k: {
            'agree': v['agree'], 'total': v['total'],
            'rate': v['agreement_rate'], 'ci': v['wilson_95ci'],
        } for k, v in dsl_cat_report['per_model'].items()},
        'time_s': round(t3e, 1),
    }

    # ── 4. herd7 Cross-Validation ──
    print("\n[4/6] herd7 Cross-Validation...")
    from herd7_validation import validate_against_herd7
    t4 = time.time()
    herd7_report = validate_against_herd7()
    t4e = time.time() - t4
    print(f"  Agreement: {herd7_report['agreements']}/{herd7_report['total_checks']} "
          f"({herd7_report['agreement_rate']*100:.1f}%)")
    print(f"  Wilson 95% CI: [{herd7_report['wilson_95ci'][0]*100:.1f}%, "
          f"{herd7_report['wilson_95ci'][1]*100:.1f}%]")
    print(f"  Completed in {t4e:.1f}s")

    with open('paper_results_v5/herd7_validation.json', 'w') as f:
        json.dump(herd7_report, f, indent=2, default=str)
    all_results['herd7_validation'] = {
        'total': herd7_report['total_checks'],
        'agree': herd7_report['agreements'],
        'rate': round(herd7_report['agreement_rate'] * 100, 1),
        'wilson_95ci': [round(x * 100, 1) for x in herd7_report['wilson_95ci']],
        'time_s': round(t4e, 1),
    }

    # ── 5. Comprehensive Fence Sufficiency Proofs ──
    print("\n[5/6] Fence Sufficiency Proofs (SMT)...")
    from smt_validation import classify_all_unsafe_pairs as smt_classify
    t5 = time.time()
    fence_report = smt_classify()
    t5e = time.time() - t5
    print(f"  Total unsafe CPU pairs: {fence_report['total_unsafe_pairs']}")
    print(f"  Fence-sufficient (UNSAT): {fence_report['fence_sufficient']}")
    print(f"  Inherently observable: {fence_report['inherently_observable']}")
    print(f"  Partial fence: {fence_report['partial_fence']}")
    print(f"  Completed in {t5e:.1f}s")

    with open('paper_results_v5/fence_proofs.json', 'w') as f:
        json.dump(fence_report, f, indent=2, default=str)
    all_results['fence_proofs'] = {
        'total_unsafe_cpu': fence_report['total_unsafe_pairs'],
        'fence_sufficient': fence_report['fence_sufficient'],
        'inherently_observable': fence_report['inherently_observable'],
        'partial_fence': fence_report['partial_fence'],
        'time_s': round(t5e, 1),
    }

    # ── 6. SMT Cross-Validation ──
    print("\n[6/6] SMT vs Enumeration Cross-Validation...")
    from smt_validation import cross_validate_smt, cross_validate_gpu_smt
    t6 = time.time()
    cpu_xval = cross_validate_smt()
    gpu_xval = cross_validate_gpu_smt()
    t6e = time.time() - t6
    print(f"  CPU: {cpu_xval['agree']}/{cpu_xval['total_checks']} agree "
          f"({cpu_xval.get('agreement_rate', 'N/A')}%)")
    print(f"  GPU: {gpu_xval['agree']}/{gpu_xval['total_checks']} agree "
          f"({gpu_xval.get('agreement_rate', 'N/A')}%)")
    print(f"  Completed in {t6e:.1f}s")

    with open('paper_results_v5/smt_cross_validation.json', 'w') as f:
        json.dump({'cpu': cpu_xval, 'gpu': gpu_xval}, f, indent=2, default=str)
    all_results['smt_cross_validation'] = {
        'cpu_agree': cpu_xval['agree'],
        'cpu_total': cpu_xval['total_checks'],
        'gpu_agree': gpu_xval['agree'],
        'gpu_total': gpu_xval['total_checks'],
        'time_s': round(t6e, 1),
    }

    # ── Portability Matrix ──
    print("\n[+] Generating full portability matrix...")
    from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest, recommend_fence
    matrix = []
    safe_count = 0
    unsafe_count = 0
    cpu_archs = ['x86', 'sparc', 'arm', 'riscv']
    gpu_archs = ['opencl_wg', 'opencl_dev', 'vulkan_wg', 'vulkan_dev', 'ptx_cta', 'ptx_gpu']
    all_archs = cpu_archs + gpu_archs

    for pat_name in sorted(PATTERNS.keys()):
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )
        for arch_name in all_archs:
            allowed, n_checked = verify_test(lt, ARCHITECTURES[arch_name])
            safe = not allowed
            fence_rec = None
            if not safe:
                fence_rec = recommend_fence(lt, arch_name, ARCHITECTURES[arch_name])
                unsafe_count += 1
            else:
                safe_count += 1

            # Look up severity
            sev = 'safe'
            if not safe:
                for sc in severity_report.get('classifications', []):
                    if sc['pattern'] == pat_name and sc['arch'] == arch_name:
                        sev = sc['severity']
                        break

            # Look up Z3 certificate
            z3_cert = 'N/A'
            for cr in cert_report.get('results', []):
                if cr['pattern'] == pat_name and cr['arch'] == arch_name:
                    z3_cert = cr.get('certificate_type', 'N/A')
                    break

            matrix.append({
                'pattern': pat_name,
                'arch': arch_name,
                'safe': safe,
                'severity': sev,
                'fence_recommendation': fence_rec,
                'z3_certificate': z3_cert,
            })

    with open('paper_results_v5/portability_matrix.json', 'w') as f:
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
    }

    # ── Summary ──
    total_time = time.time() - t0
    all_results['total_time_s'] = round(total_time, 1)

    with open('paper_results_v5/experiment_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Z3 Certificate Coverage: {all_results['universal_certificates']['certified']}/"
          f"{all_results['universal_certificates']['total_pairs']} "
          f"({all_results['universal_certificates']['coverage_pct']}%)")
    print(f"Severity Classification: {all_results['severity_classification']['total_unsafe']} unsafe pairs classified")
    for sev, count in sorted(all_results['severity_classification']['severity_counts'].items()):
        print(f"  {sev}: {count}")
    print(f"DSL-.cat Correspondence: {all_results['dsl_cat_correspondence']['agree']}/"
          f"{all_results['dsl_cat_correspondence']['total_checks']} "
          f"({all_results['dsl_cat_correspondence']['agreement_rate']}%)")
    print(f"herd7 Validation: {all_results['herd7_validation']['agree']}/"
          f"{all_results['herd7_validation']['total']} ({all_results['herd7_validation']['rate']}%)")
    print(f"Portability Matrix: {all_results['portability_matrix']['safe']} safe, "
          f"{all_results['portability_matrix']['unsafe']} unsafe")
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"All results saved to paper_results_v5/")

    return all_results


if __name__ == '__main__':
    run_all_phase_b_experiments()
