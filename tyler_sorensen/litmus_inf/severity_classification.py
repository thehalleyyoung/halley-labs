#!/usr/bin/env python3
"""
Severity classification for unsafe (pattern, architecture) pairs in LITMUS∞.

Classifies each unsafe pair into one of four severity levels:
  - data_race: Concurrent access to shared memory without proper synchronization,
    leading to incorrect reads (e.g., message passing without fences).
  - security_vulnerability: Data races that can leak information or break
    mutual exclusion invariants (e.g., Dekker, Peterson without fences).
  - performance_only: Pattern involves redundant or suboptimal fencing —
    safe but with unnecessary overhead.
  - benign: Coherence-order artifacts that are architecturally allowed but
    do not affect correctness for most programs (e.g., CoWR, CoWW observers).

Classification is based on:
  1. Pattern structure (which memory ordering is violated)
  2. Whether a fenced variant exists and restores safety
  3. Pattern category (mutual exclusion, message passing, etc.)
  4. Whether the violation involves data integrity or just ordering visibility
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest, HERD7_EXPECTED
from smt_validation import validate_pattern_smt, validate_gpu_pattern_smt, Z3_AVAILABLE
from statistical_analysis import wilson_ci

# Pattern classification rules
PATTERN_SEVERITY_RULES = {
    # Mutual exclusion patterns — security vulnerability
    'dekker': 'security_vulnerability',
    'peterson': 'security_vulnerability',

    # Message passing — data race (stale/torn reads)
    'mp': 'data_race',
    'mp_3thread': 'data_race',
    'mp_addr': 'data_race',
    'mp_co': 'data_race',
    'mp_data': 'data_race',
    'mp_rfi': 'data_race',

    # Store buffering — data race (missed updates)
    'sb': 'data_race',
    'sb_3thread': 'data_race',
    'sb_rfi': 'data_race',

    # Load buffering — data race (out-of-thin-air values)
    'lb': 'data_race',
    'lb_data': 'data_race',

    # Multi-copy atomicity — data race (inconsistent global ordering)
    'iriw': 'data_race',

    # Causality patterns — data race
    'isa2': 'data_race',
    'r': 'data_race',
    'wrc': 'data_race',
    'wrc_addr': 'data_race',
    'rwc': 'data_race',
    's': 'data_race',
    '3sb': 'data_race',

    # Write ordering — benign (coherence visibility)
    '2+2w': 'benign',
    'cowr': 'benign',
    'coww': 'benign',

    # Fenced variants of above — if still unsafe, it's an insufficient fence
    'mp_dmb_ld': 'data_race',    # partial fence (R->R only)
    'mp_dmb_st': 'data_race',    # partial fence (W->W only)
    'mp_fence_wr': 'data_race',  # wrong fence direction

    # Fenced variants that should be safe — if unsafe, it's a scope/model issue
    '2+2w_fence': 'benign',
    'cowr_fence': 'benign',
    'coww_fence': 'benign',
    'mp_3thread_fence': 'data_race',
    's_fence': 'benign',

    # GPU patterns
    'gpu_2plus2w_xwg': 'data_race',
    'gpu_barrier_scope_mismatch': 'security_vulnerability',
    'gpu_iriw_dev': 'data_race',
    'gpu_iriw_scope_mismatch': 'security_vulnerability',
    'gpu_mp_dev': 'data_race',
    'gpu_mp_scope_mismatch_dev': 'security_vulnerability',
    'gpu_release_acquire': 'data_race',
    'gpu_rwc_dev': 'data_race',
    'gpu_sb_dev': 'data_race',
    'gpu_sb_scope_mismatch': 'security_vulnerability',
    'gpu_wrc_dev': 'data_race',
}

# CWE mapping for severity categories
# Maps each severity level to relevant CWE entries for calibration
SEVERITY_CWE_MAPPING = {
    'data_race': {
        'primary': 'CWE-362',
        'cwe_name': 'Concurrent Execution using Shared Resource with Improper Synchronization (Race Condition)',
        'related': ['CWE-366', 'CWE-367', 'CWE-820'],
        'related_names': [
            'CWE-366: Race Condition within a Thread',
            'CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition',
            'CWE-820: Missing Synchronization',
        ],
        'note': 'Weak-memory data races are a subset of CWE-362 where the race arises from architecture-specific memory reordering rather than missing locks.',
    },
    'security_vulnerability': {
        'primary': 'CWE-667',
        'cwe_name': 'Improper Locking',
        'related': ['CWE-362', 'CWE-764', 'CWE-821'],
        'related_names': [
            'CWE-362: Race Condition',
            'CWE-764: Multiple Locks of a Critical Resource',
            'CWE-821: Incorrect Synchronization',
        ],
        'note': 'Patterns like Dekker/Peterson implement mutual exclusion that breaks under relaxed memory models, mapping to CWE-667 (improper locking) since the synchronization mechanism becomes unsound.',
    },
    'benign': {
        'primary': None,
        'cwe_name': 'No direct CWE mapping',
        'related': [],
        'related_names': [],
        'note': 'Coherence-order artifacts (CoWR, CoWW, 2+2W) are architecturally observable but do not correspond to security vulnerabilities in typical single-writer programs.',
    },
}


def _get_base_pattern(pat_name):
    """Strip fence suffixes to get the base pattern name."""
    for suffix in ['_fence_ww_rr', '_fence_wr', '_fence_rw',
                   '_dmb_ld', '_dmb_st', '_fence', '_data_fence',
                   '_addr_fence', '_co_fence', '_rfi_fence']:
        if pat_name.endswith(suffix):
            return pat_name[:-len(suffix)]
    return pat_name


def _has_fenced_variant(pat_name):
    """Check if this pattern has a fenced variant that is safe on some arch."""
    fenced = pat_name + '_fence'
    return fenced in PATTERNS


def classify_severity(pat_name, arch_name, enum_allowed):
    """Classify the severity of a single unsafe pair.

    Args:
        pat_name: Pattern name
        arch_name: Architecture name
        enum_allowed: Whether the forbidden outcome is allowed (True = unsafe)

    Returns:
        dict with severity, rationale, and metadata
    """
    if not enum_allowed:
        return {
            'severity': 'safe',
            'rationale': 'Forbidden outcome is not observable'
        }

    # Look up base severity from rules
    base_pat = _get_base_pattern(pat_name)
    severity = PATTERN_SEVERITY_RULES.get(pat_name,
               PATTERN_SEVERITY_RULES.get(base_pat, 'data_race'))

    # Determine if a fenced version restores safety
    fenced_name = pat_name + '_fence'
    fenceable = fenced_name in PATTERNS
    fence_fixes = False
    if fenceable:
        pat_def = PATTERNS[fenced_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=fenced_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )
        fence_allowed, _ = verify_test(lt, ARCHITECTURES[arch_name])
        fence_fixes = not fence_allowed

    # Determine violated ordering pairs
    pat_def = PATTERNS[pat_name]
    ops = pat_def['ops']
    violated_types = set()
    from collections import defaultdict
    ops_by_thread = defaultdict(list)
    for op in ops:
        if op.optype != 'fence':
            ops_by_thread[op.thread].append(op)
    for t, t_ops in ops_by_thread.items():
        for i in range(len(t_ops)):
            for j in range(i + 1, len(t_ops)):
                a, b = t_ops[i], t_ops[j]
                if a.addr != b.addr:
                    violated_types.add(f"{a.optype}->{b.optype}")

    # Build rationale
    if severity == 'security_vulnerability':
        rationale = (f"Pattern {pat_name} implements mutual exclusion; "
                    f"violation breaks atomicity guarantees on {arch_name}")
    elif severity == 'data_race':
        pairs_str = ', '.join(sorted(violated_types))
        rationale = (f"Relaxed ordering ({pairs_str}) on {arch_name} allows "
                    f"stale/inconsistent reads")
        if fenceable and fence_fixes:
            rationale += "; fixable with appropriate fences"
        elif fenceable and not fence_fixes:
            rationale += "; fences insufficient (architecture-level limitation)"
    elif severity == 'benign':
        rationale = (f"Coherence ordering artifact on {arch_name}; "
                    f"does not affect data integrity for most programs")
    else:
        rationale = f"Unclassified unsafe behavior on {arch_name}"

    return {
        'severity': severity,
        'rationale': rationale,
        'fenceable': fenceable,
        'fence_fixes': fence_fixes,
        'violated_orderings': sorted(violated_types),
        'cwe_mapping': SEVERITY_CWE_MAPPING.get(severity, {}),
    }


def classify_all_unsafe_pairs():
    """Classify all 342 unsafe pairs by severity.

    Returns comprehensive classification report with per-severity counts,
    per-architecture breakdown, and Z3 certificate cross-references.
    """
    cpu_archs = ['x86', 'sparc', 'arm', 'riscv']
    gpu_archs = ['opencl_wg', 'opencl_dev', 'vulkan_wg', 'vulkan_dev',
                 'ptx_cta', 'ptx_gpu']
    all_archs = cpu_archs + gpu_archs

    results = []
    severity_counts = {'data_race': 0, 'security_vulnerability': 0,
                       'performance_only': 0, 'benign': 0}
    arch_severity = {}

    for pat_name in sorted(PATTERNS.keys()):
        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )

        for arch_name in all_archs:
            enum_allowed, _ = verify_test(lt, ARCHITECTURES[arch_name])
            if not enum_allowed:
                continue  # Safe pair

            classification = classify_severity(pat_name, arch_name, enum_allowed)

            # Z3 certificate
            cpu_model_map = {'x86': 'TSO', 'sparc': 'PSO', 'arm': 'ARM', 'riscv': 'RISC-V'}
            if arch_name in cpu_archs:
                smt = validate_pattern_smt(pat_name, cpu_model_map[arch_name])
            else:
                smt = validate_gpu_pattern_smt(pat_name, arch_name)

            entry = {
                'pattern': pat_name,
                'arch': arch_name,
                'description': pat_def.get('description', ''),
                'severity': classification['severity'],
                'rationale': classification['rationale'],
                'fenceable': classification['fenceable'],
                'fence_fixes': classification['fence_fixes'],
                'violated_orderings': classification['violated_orderings'],
                'z3_certificate': smt.get('smt_result', 'N/A'),
                'z3_time_ms': smt.get('time_ms', 0),
            }
            results.append(entry)

            sev = classification['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if arch_name not in arch_severity:
                arch_severity[arch_name] = {'data_race': 0, 'security_vulnerability': 0,
                                             'performance_only': 0, 'benign': 0}
            arch_severity[arch_name][sev] = arch_severity[arch_name].get(sev, 0) + 1

    # Compute stats
    total_unsafe = len(results)
    certified = sum(1 for r in results if r['z3_certificate'] in ('sat', 'unsat'))
    fenceable_count = sum(1 for r in results if r['fenceable'])
    fence_fixes_count = sum(1 for r in results if r['fence_fixes'])

    report = {
        'total_unsafe_pairs': total_unsafe,
        'severity_counts': severity_counts,
        'per_architecture': arch_severity,
        'z3_certified': certified,
        'fenceable': fenceable_count,
        'fence_fixes': fence_fixes_count,
        'classifications': results,
    }

    return report


def run_severity_classification():
    """Run full severity classification and save results."""
    print("=" * 70)
    print("LITMUS∞ Severity Classification of Unsafe Pairs")
    print("=" * 70)

    start = time.time()
    report = classify_all_unsafe_pairs()
    elapsed = time.time() - start

    print(f"\nTotal unsafe pairs: {report['total_unsafe_pairs']}")
    print(f"\nSeverity breakdown:")
    for sev, count in sorted(report['severity_counts'].items()):
        pct = 100 * count / max(report['total_unsafe_pairs'], 1)
        print(f"  {sev}: {count} ({pct:.1f}%)")

    print(f"\nPer-architecture severity:")
    for arch in sorted(report['per_architecture'].keys()):
        counts = report['per_architecture'][arch]
        total = sum(counts.values())
        print(f"  {arch} ({total} unsafe): ", end='')
        parts = [f"{sev}={c}" for sev, c in sorted(counts.items()) if c > 0]
        print(', '.join(parts))

    print(f"\nZ3 certified: {report['z3_certified']}/{report['total_unsafe_pairs']}")
    print(f"Fenceable: {report['fenceable']}, Fence fixes: {report['fence_fixes']}")
    print(f"\nCompleted in {elapsed:.1f}s")

    os.makedirs('paper_results_v5', exist_ok=True)
    with open('paper_results_v5/severity_classification.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved to paper_results_v5/severity_classification.json")

    return report


if __name__ == '__main__':
    run_severity_classification()
