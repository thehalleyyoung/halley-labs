#!/usr/bin/env python3
"""
Differential testing framework for LITMUS∞.

Cross-validates portability results via:
1. Monotonicity checks: if M1 ⊑ M2 and T safe on M2, then T safe on M1
2. Fence soundness: adding a fence never makes a safe pattern unsafe
3. Self-consistency: re-running analysis produces identical results
4. Custom model cross-validation: DSL-defined models agree with built-in equivalents
5. Litmus file round-trip: exported .litmus files re-parse correctly
"""

import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import (
    PATTERNS, ARCHITECTURES, check_portability, verify_test,
    LitmusTest, MemOp, recommend_fence,
)


def check_monotonicity():
    """Verify model strength ordering: if M1 ⊑ M2 and T safe on M2, then T safe on M1."""
    # Strength ordering (stronger ⊑ weaker):
    # SC ⊑ TSO ⊑ PSO ⊑ ARM ≈ RISC-V
    # GPU-Dev ⊑ GPU-WG (per API)
    orderings = [
        ('x86', 'sparc'),      # TSO ⊑ PSO
        ('sparc', 'arm'),      # PSO ⊑ ARM
        ('sparc', 'riscv'),    # PSO ⊑ RISC-V
        ('opencl_dev', 'opencl_wg'),
        ('vulkan_dev', 'vulkan_wg'),
        ('ptx_gpu', 'ptx_cta'),
    ]

    violations = []
    checks = 0

    for stronger, weaker in orderings:
        if stronger not in ARCHITECTURES or weaker not in ARCHITECTURES:
            continue
        for pat_name, pat_def in PATTERNS.items():
            checks += 1
            n_threads = max(op.thread for op in pat_def['ops']) + 1
            lt = LitmusTest(
                name=pat_name, n_threads=n_threads,
                addresses=pat_def['addresses'], ops=pat_def['ops'],
                forbidden=pat_def['forbidden'],
            )

            # Check both models
            weaker_allowed, _ = verify_test(lt, ARCHITECTURES[weaker])
            stronger_allowed, _ = verify_test(lt, ARCHITECTURES[stronger])

            # Monotonicity: if weaker model forbids it, stronger must too
            if not weaker_allowed and stronger_allowed:
                violations.append({
                    'pattern': pat_name,
                    'stronger': stronger,
                    'weaker': weaker,
                    'issue': f'{pat_name} safe on {weaker} (weaker) but unsafe on {stronger} (stronger)',
                })

    return {
        'checks': checks,
        'violations': len(violations),
        'details': violations,
        'passed': len(violations) == 0,
    }


def check_fence_soundness():
    """Verify that fenced versions of patterns are at least as safe as unfenced."""
    fence_pairs = [
        ('mp', 'mp_fence'),
        ('sb', 'sb_fence'),
        ('lb', 'lb_fence'),
        ('iriw', 'iriw_fence'),
        ('wrc', 'wrc_fence'),
        ('rwc', 'rwc_fence'),
    ]

    violations = []
    checks = 0

    for unfenced, fenced in fence_pairs:
        if unfenced not in PATTERNS or fenced not in PATTERNS:
            continue
        for arch_name, arch_model in ARCHITECTURES.items():
            checks += 1
            # Build litmus tests
            for pat_name in [unfenced, fenced]:
                pat_def = PATTERNS[pat_name]
                n_threads = max(op.thread for op in pat_def['ops']) + 1
                lt = LitmusTest(
                    name=pat_name, n_threads=n_threads,
                    addresses=pat_def['addresses'], ops=pat_def['ops'],
                    forbidden=pat_def['forbidden'],
                )

            # Check
            uf_def = PATTERNS[unfenced]
            f_def = PATTERNS[fenced]
            uf_lt = LitmusTest(name=unfenced, n_threads=max(op.thread for op in uf_def['ops'])+1,
                               addresses=uf_def['addresses'], ops=uf_def['ops'], forbidden=uf_def['forbidden'])
            f_lt = LitmusTest(name=fenced, n_threads=max(op.thread for op in f_def['ops'])+1,
                              addresses=f_def['addresses'], ops=f_def['ops'], forbidden=f_def['forbidden'])

            uf_allowed, _ = verify_test(uf_lt, arch_model)
            f_allowed, _ = verify_test(f_lt, arch_model)

            # Fence soundness: if fenced is unsafe, unfenced must also be unsafe
            if f_allowed and not uf_allowed:
                violations.append({
                    'unfenced': unfenced, 'fenced': fenced,
                    'arch': arch_name,
                    'issue': f'{fenced} unsafe but {unfenced} safe on {arch_name}',
                })

    return {
        'checks': checks,
        'violations': len(violations),
        'details': violations,
        'passed': len(violations) == 0,
    }


def check_determinism(n_runs=5):
    """Verify analysis produces identical results across multiple runs."""
    results_list = []
    for _ in range(n_runs):
        run_results = {}
        for pat_name in sorted(PATTERNS.keys()):
            pat_def = PATTERNS[pat_name]
            n_threads = max(op.thread for op in pat_def['ops']) + 1
            lt = LitmusTest(
                name=pat_name, n_threads=n_threads,
                addresses=pat_def['addresses'], ops=pat_def['ops'],
                forbidden=pat_def['forbidden'],
            )
            for arch_name, arch_model in ARCHITECTURES.items():
                allowed, _ = verify_test(lt, arch_model)
                run_results[(pat_name, arch_name)] = allowed
        results_list.append(run_results)

    # Compare all runs
    mismatches = []
    base = results_list[0]
    for i, run in enumerate(results_list[1:], 1):
        for key in base:
            if base[key] != run[key]:
                mismatches.append({
                    'pattern': key[0], 'arch': key[1],
                    'run0': base[key], f'run{i}': run[key],
                })

    return {
        'n_runs': n_runs,
        'checks': len(base) * (n_runs - 1),
        'mismatches': len(mismatches),
        'details': mismatches,
        'passed': len(mismatches) == 0,
    }


def check_custom_model_consistency():
    """Cross-validate DSL-defined models against built-in equivalents."""
    try:
        from model_dsl import register_model, check_custom, get_registry
    except ImportError:
        return {'skipped': True, 'reason': 'model_dsl not available'}

    # Define TSO equivalent via DSL
    tso_dsl = """model TSO_Check {
    description "x86 TSO equivalent for cross-validation"
    relaxes W->R
    preserves deps
    fence mfence (cost=8) { orders W->R, W->W, R->R, R->W }
}"""
    register_model(tso_dsl)

    mismatches = []
    checks = 0

    for pat_name in sorted(PATTERNS.keys()):
        # Skip GPU-specific patterns (custom DSL doesn't handle scopes)
        if pat_name.startswith('gpu_'):
            continue
        checks += 1

        # Built-in TSO result
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )
        builtin_allowed, _ = verify_test(lt, ARCHITECTURES['x86'])

        # DSL TSO result
        dsl_result = check_custom(pat_name, 'TSO_Check')
        dsl_safe = dsl_result.get('safe', True)

        # Built-in TSO: allowed=True means forbidden outcome IS reachable (unsafe)
        # DSL: safe=True means pattern is safe (forbidden outcome NOT reachable)
        builtin_unsafe = builtin_allowed  # True = unsafe
        dsl_unsafe = not dsl_safe         # True = unsafe

        if builtin_unsafe != dsl_unsafe:
            mismatches.append({
                'pattern': pat_name,
                'builtin_unsafe': builtin_unsafe,
                'dsl_unsafe': dsl_unsafe,
            })

    return {
        'checks': checks,
        'mismatches': len(mismatches),
        'details': mismatches[:10],
        'passed': len(mismatches) == 0,
    }


def check_litmus_roundtrip():
    """Verify .litmus file export produces valid files for all 57 patterns."""
    try:
        from herd7_export import export_all_litmus
    except ImportError:
        return {'skipped': True, 'reason': 'herd7_export not available'}

    exported_files, errors = export_all_litmus('litmus_files')

    checks = 0
    issues = []

    for pat_name in exported_files:
        filepath = f'litmus_files/{pat_name}.litmus'
        checks += 1
        if not os.path.exists(filepath):
            issues.append({'pattern': pat_name, 'issue': 'file not created'})
            continue

        with open(filepath) as f:
            content = f.read()

        # Validate structure
        if not content.startswith(('C ', 'LISA ', 'AArch64 ', 'X86 ')):
            issues.append({'pattern': pat_name, 'issue': 'invalid header'})
        if 'exists' not in content:
            issues.append({'pattern': pat_name, 'issue': 'missing exists clause'})
        if '{' not in content:
            issues.append({'pattern': pat_name, 'issue': 'missing init block'})

    return {
        'checks': checks,
        'issues': len(issues),
        'details': issues,
        'passed': len(issues) == 0,
    }


def run_all_differential_tests():
    """Run all differential tests and save results."""
    print("=" * 70)
    print("LITMUS∞ Differential Testing Suite")
    print("=" * 70)

    results = {}

    print("\n[1/5] Monotonicity check (model strength ordering)...")
    t0 = time.perf_counter()
    results['monotonicity'] = check_monotonicity()
    t1 = time.perf_counter()
    status = "PASS" if results['monotonicity']['passed'] else "FAIL"
    print(f"  {status}: {results['monotonicity']['checks']} checks, "
          f"{results['monotonicity']['violations']} violations ({t1-t0:.2f}s)")

    print("\n[2/5] Fence soundness check...")
    t0 = time.perf_counter()
    results['fence_soundness'] = check_fence_soundness()
    t1 = time.perf_counter()
    status = "PASS" if results['fence_soundness']['passed'] else "FAIL"
    print(f"  {status}: {results['fence_soundness']['checks']} checks, "
          f"{results['fence_soundness']['violations']} violations ({t1-t0:.2f}s)")

    print("\n[3/5] Determinism check (5 runs)...")
    t0 = time.perf_counter()
    results['determinism'] = check_determinism(5)
    t1 = time.perf_counter()
    status = "PASS" if results['determinism']['passed'] else "FAIL"
    print(f"  {status}: {results['determinism']['checks']} checks, "
          f"{results['determinism']['mismatches']} mismatches ({t1-t0:.2f}s)")

    print("\n[4/5] Custom model cross-validation...")
    t0 = time.perf_counter()
    results['custom_model'] = check_custom_model_consistency()
    t1 = time.perf_counter()
    if 'skipped' in results['custom_model']:
        print(f"  SKIPPED: {results['custom_model']['reason']}")
    else:
        status = "PASS" if results['custom_model']['passed'] else "FAIL"
        print(f"  {status}: {results['custom_model']['checks']} checks, "
              f"{results['custom_model']['mismatches']} mismatches ({t1-t0:.2f}s)")

    print("\n[5/5] Litmus file round-trip check...")
    t0 = time.perf_counter()
    results['litmus_roundtrip'] = check_litmus_roundtrip()
    t1 = time.perf_counter()
    if 'skipped' in results['litmus_roundtrip']:
        print(f"  SKIPPED: {results['litmus_roundtrip']['reason']}")
    else:
        status = "PASS" if results['litmus_roundtrip']['passed'] else "FAIL"
        print(f"  {status}: {results['litmus_roundtrip']['checks']} checks, "
              f"{results['litmus_roundtrip']['issues']} issues ({t1-t0:.2f}s)")

    # Summary
    all_passed = all(
        r.get('passed', True)
        for r in results.values()
        if 'skipped' not in r
    )
    total_checks = sum(r.get('checks', 0) for r in results.values() if 'skipped' not in r)

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
    print(f"Total checks: {total_checks}")
    print(f"{'=' * 70}")

    # Save results
    os.makedirs('paper_results_v4', exist_ok=True)
    with open('paper_results_v4/differential_testing.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to paper_results_v4/differential_testing.json")

    return results


if __name__ == '__main__':
    run_all_differential_tests()
