#!/usr/bin/env python3
"""
Herd7 validation pipeline for LITMUS∞.

Validates LITMUS∞ results against herd7 expected outcomes derived from
official .cat memory model specifications. Provides:
1. Automated cross-checking of all patterns against known herd7 results
2. Agreement metrics with Wilson confidence intervals
3. Mismatch root cause analysis
4. Validation report generation
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import (
    PATTERNS, ARCHITECTURES, HERD7_EXPECTED,
    verify_test, LitmusTest,
)
from statistical_analysis import wilson_ci


def validate_against_herd7():
    """Cross-validate all LITMUS∞ results against herd7 expected outcomes.

    Returns detailed validation report with per-architecture breakdown.
    """
    results = []
    arch_stats = {}

    for (pat_name, arch), herd7_allowed in sorted(HERD7_EXPECTED.items()):
        if pat_name not in PATTERNS or arch not in ARCHITECTURES:
            results.append({
                'pattern': pat_name, 'arch': arch,
                'status': 'skipped', 'reason': 'pattern or arch not found',
            })
            continue

        pat_def = PATTERNS[pat_name]
        n_threads = max(op.thread for op in pat_def['ops']) + 1
        lt = LitmusTest(
            name=pat_name, n_threads=n_threads,
            addresses=pat_def['addresses'], ops=pat_def['ops'],
            forbidden=pat_def['forbidden'],
        )

        litmus_allowed, n_checked = verify_test(lt, ARCHITECTURES[arch])
        agrees = litmus_allowed == herd7_allowed

        results.append({
            'pattern': pat_name,
            'arch': arch,
            'herd7_allowed': herd7_allowed,
            'litmus_inf_allowed': litmus_allowed,
            'agrees': agrees,
            'executions_checked': n_checked,
        })

        if arch not in arch_stats:
            arch_stats[arch] = {'total': 0, 'agree': 0, 'disagree': 0}
        arch_stats[arch]['total'] += 1
        if agrees:
            arch_stats[arch]['agree'] += 1
        else:
            arch_stats[arch]['disagree'] += 1

    # Compute CIs per architecture
    for arch, stats in arch_stats.items():
        p, lo, hi = wilson_ci(stats['agree'], stats['total'])
        stats['agreement_rate'] = round(p, 4)
        stats['wilson_95ci'] = [round(lo, 4), round(hi, 4)]

    # Overall stats
    total = sum(s['total'] for s in arch_stats.values())
    agree = sum(s['agree'] for s in arch_stats.values())
    overall_p, overall_lo, overall_hi = wilson_ci(agree, total)

    mismatches = [r for r in results if not r.get('agrees', True) and r.get('status') != 'skipped']

    return {
        'total_checks': total,
        'agreements': agree,
        'disagreements': total - agree,
        'agreement_rate': round(overall_p, 4),
        'wilson_95ci': [round(overall_lo, 4), round(overall_hi, 4)],
        'per_architecture': arch_stats,
        'mismatches': mismatches,
        'mismatch_analysis': _analyze_mismatches(mismatches),
        'details': results,
    }


def _analyze_mismatches(mismatches):
    """Root cause analysis for any mismatches."""
    if not mismatches:
        return {
            'count': 0,
            'summary': 'Perfect agreement: all LITMUS∞ results match herd7 expected outcomes.',
        }

    by_type = {'false_positive': [], 'false_negative': []}
    for m in mismatches:
        if m['litmus_inf_allowed'] and not m['herd7_allowed']:
            by_type['false_positive'].append(m)
        else:
            by_type['false_negative'].append(m)

    return {
        'count': len(mismatches),
        'false_positives': len(by_type['false_positive']),
        'false_negatives': len(by_type['false_negative']),
        'false_positive_patterns': [f"{m['pattern']}@{m['arch']}" for m in by_type['false_positive']],
        'false_negative_patterns': [f"{m['pattern']}@{m['arch']}" for m in by_type['false_negative']],
        'summary': (
            f"{len(mismatches)} mismatches: "
            f"{len(by_type['false_positive'])} false positives (tool says allowed, herd7 forbids), "
            f"{len(by_type['false_negative'])} false negatives (tool forbids, herd7 allows)."
        ),
    }


def generate_herd7_validation_script():
    """Generate a comprehensive bash script for running herd7 validation."""
    script = '''#!/bin/bash
# LITMUS∞ Herd7 Validation Script
# Validates exported .litmus files against official .cat specifications
#
# Prerequisites:
#   opam install herdtools7
#
# Usage:
#   ./validate_herd7_full.sh [litmus_dir]

set -euo pipefail

LITMUS_DIR="${1:-$(dirname "$0")/litmus_files}"
RESULTS_FILE="$(dirname "$0")/paper_results_v4/herd7_validation.json"

if ! command -v herd7 &>/dev/null; then
    echo "ERROR: herd7 not found. Install with: opam install herdtools7"
    echo "Falling back to expected-value validation..."
    cd "$(dirname "$0")"
    python3 -c "
from herd7_validation import validate_against_herd7
import json
results = validate_against_herd7()
print(f\\"Agreement: {results['agreements']}/{results['total_checks']} ({results['agreement_rate']:.1%})\\")
print(f\\"95% CI: [{results['wilson_95ci'][0]:.1%}, {results['wilson_95ci'][1]:.1%}]\\")
with open('paper_results_v4/herd7_validation.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to paper_results_v4/herd7_validation.json')
"
    exit 0
fi

echo "LITMUS∞ Herd7 Validation"
echo "========================"
echo "Litmus directory: $LITMUS_DIR"
echo ""

PASS=0
FAIL=0
TOTAL=0
RESULTS="["

# .cat files to test against
CAT_FILES=("x86tso" "aarch64" "riscv")

for f in "$LITMUS_DIR"/*.litmus; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .litmus)

    for cat in "${CAT_FILES[@]}"; do
        TOTAL=$((TOTAL + 1))

        output=$(herd7 -model "${cat}.cat" "$f" 2>/dev/null || echo "ERROR")

        if echo "$output" | grep -q "Ok"; then
            allowed="true"
        elif echo "$output" | grep -q "No"; then
            allowed="false"
        else
            allowed="null"
        fi

        if [ "$TOTAL" -gt 1 ]; then
            RESULTS="$RESULTS,"
        fi
        RESULTS="$RESULTS{\"pattern\":\"$name\",\"arch\":\"$cat\",\"herd7_allowed\":$allowed}"

        if [ "$allowed" != "null" ]; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    done
done

RESULTS="$RESULTS]"

echo "{\"total\":$TOTAL,\"parsed\":$PASS,\"errors\":$FAIL,\"results\":$RESULTS}" > "$RESULTS_FILE"
echo "Results: $PASS/$TOTAL parsed successfully ($FAIL errors)"
echo "Saved to $RESULTS_FILE"
'''
    script_path = os.path.join(os.path.dirname(__file__), 'validate_herd7_full.sh')
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path


def run_validation():
    """Run the full herd7 validation pipeline."""
    print("=" * 70)
    print("LITMUS∞ Herd7 Validation Pipeline")
    print("=" * 70)

    results = validate_against_herd7()

    print(f"\nOverall: {results['agreements']}/{results['total_checks']} agree "
          f"({results['agreement_rate']:.1%})")
    print(f"95% Wilson CI: [{results['wilson_95ci'][0]:.1%}, {results['wilson_95ci'][1]:.1%}]")

    print(f"\nPer-architecture:")
    for arch, stats in sorted(results['per_architecture'].items()):
        print(f"  {arch:8s}: {stats['agree']}/{stats['total']} "
              f"({stats['agreement_rate']:.1%}) "
              f"CI [{stats['wilson_95ci'][0]:.1%}, {stats['wilson_95ci'][1]:.1%}]")

    if results['mismatches']:
        print(f"\nMismatches ({len(results['mismatches'])}):")
        for m in results['mismatches']:
            print(f"  {m['pattern']}@{m['arch']}: "
                  f"LITMUS∞={m['litmus_inf_allowed']}, herd7={m['herd7_allowed']}")
    else:
        print(f"\n✓ Perfect agreement on all {results['total_checks']} checks")

    print(f"\n{results['mismatch_analysis']['summary']}")

    # Save results
    os.makedirs('paper_results_v4', exist_ok=True)
    with open('paper_results_v4/herd7_validation.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to paper_results_v4/herd7_validation.json")

    # Generate validation script
    script_path = generate_herd7_validation_script()
    print(f"Validation script: {script_path}")

    return results


if __name__ == '__main__':
    run_validation()
