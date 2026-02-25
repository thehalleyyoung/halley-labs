#!/usr/bin/env python3
"""
False-negative analysis for LITMUS∞ AST-based code analyzer.

Classifies ALL non-exact-match cases as:
- SAFE (conservative): predicted pattern is at least as strict as expected
  → tool would flag potential issue, no real bug missed
- NEUTRAL: predicted and expected patterns have identical portability profiles
  → no safety consequence
- UNSAFE (missed bug): predicted pattern is less strict than expected
  → tool might fail to flag an actual portability bug

This analysis directly addresses the reviewer concern about whether the
96.6% exact-match accuracy hides false negatives that could miss real bugs.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import PATTERNS, ARCHITECTURES, verify_test, LitmusTest


def get_portability_profile(pat_name):
    """Get the portability profile of a pattern across all CPU architectures."""
    if pat_name not in PATTERNS:
        return None
    pat_def = PATTERNS[pat_name]
    n_threads = max(op.thread for op in pat_def['ops']) + 1
    lt = LitmusTest(
        name=pat_name, n_threads=n_threads,
        addresses=pat_def['addresses'], ops=pat_def['ops'],
        forbidden=pat_def['forbidden'],
    )
    profile = {}
    for arch_name in ['x86', 'sparc', 'arm', 'riscv']:
        allowed, _ = verify_test(lt, ARCHITECTURES[arch_name])
        profile[arch_name] = 'unsafe' if allowed else 'safe'
    return profile


def classify_near_miss(expected_pat, predicted_pat):
    """Classify a near-miss as safe, neutral, or unsafe.

    SAFE: predicted is at least as conservative (flags ≥ issues)
    NEUTRAL: identical portability profile
    UNSAFE: predicted misses issues the expected pattern catches
    """
    expected_profile = get_portability_profile(expected_pat)
    predicted_profile = get_portability_profile(predicted_pat)

    if expected_profile is None or predicted_profile is None:
        return 'unknown', 'pattern not found', {}, {}

    # Count differences
    missed_unsafe = 0  # expected=unsafe, predicted=safe → missed bug
    extra_unsafe = 0   # expected=safe, predicted=unsafe → false positive (conservative)
    agreement = 0

    details = {}
    for arch in ['x86', 'sparc', 'arm', 'riscv']:
        e = expected_profile[arch]
        p = predicted_profile[arch]
        if e == p:
            agreement += 1
            details[arch] = 'agree'
        elif e == 'unsafe' and p == 'safe':
            missed_unsafe += 1
            details[arch] = 'MISSED (expected unsafe, predicted safe)'
        elif e == 'safe' and p == 'unsafe':
            extra_unsafe += 1
            details[arch] = 'extra flag (expected safe, predicted unsafe)'

    if missed_unsafe > 0:
        category = 'UNSAFE'
        reason = f'Misses {missed_unsafe} unsafe architecture(s)'
    elif extra_unsafe > 0:
        category = 'SAFE'
        reason = f'Conservative: flags {extra_unsafe} extra architecture(s)'
    else:
        category = 'NEUTRAL'
        reason = 'Identical portability profile'

    return category, reason, expected_profile, predicted_profile, details


def run_false_negative_analysis():
    """Analyze all non-exact-match cases from the benchmark."""
    print("=" * 70)
    print("LITMUS∞ False-Negative Analysis")
    print("Classification of All Non-Exact-Match Cases")
    print("=" * 70)
    print()

    # Load benchmark results
    bench_path = os.path.join(os.path.dirname(__file__),
                              'paper_results_v4/ast_benchmark_results.json')
    if not os.path.exists(bench_path):
        # Fall back to expanded_benchmark.json
        bench_path = os.path.join(os.path.dirname(__file__),
                                  'paper_results_v4/expanded_benchmark.json')
    if not os.path.exists(bench_path):
        print("ERROR: benchmark results not found")
        return None

    with open(bench_path) as f:
        data = json.load(f)

    near_misses = [r for r in data['results']
                   if not r['exact_match']]

    safe_count = 0
    neutral_count = 0
    unsafe_count = 0
    unknown_count = 0
    results = []

    for nm in near_misses:
        expected = nm['expected']
        predicted = nm['predicted']
        snippet_id = nm['id']

        result = classify_near_miss(expected, predicted)
        if len(result) == 4:
            category, reason, exp_prof, pred_prof = result
            details = {}
        else:
            category, reason, exp_prof, pred_prof, details = result

        if category == 'SAFE':
            safe_count += 1
            icon = '✓'
        elif category == 'NEUTRAL':
            neutral_count += 1
            icon = '='
        elif category == 'UNSAFE':
            unsafe_count += 1
            icon = '✗'
        else:
            unknown_count += 1
            icon = '?'

        print(f"  {icon} [{category}] {snippet_id}")
        print(f"    Expected: {expected} → Predicted: {predicted}")
        print(f"    Reason: {reason}")
        if details:
            for arch, detail in details.items():
                if detail != 'agree':
                    print(f"    {arch}: {detail}")
        print()

        results.append({
            'id': snippet_id,
            'expected': expected,
            'predicted': predicted,
            'category': category,
            'reason': reason,
            'expected_profile': exp_prof,
            'predicted_profile': pred_prof,
            'arch_details': details,
            'confidence': nm.get('confidence'),
            'top3': nm.get('top3', []),
        })

    # Summary
    total = len(near_misses)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total near-miss cases: {total}")
    print(f"  ✓ SAFE (conservative):    {safe_count}/{total} "
          f"({100*safe_count/max(total,1):.1f}%)")
    print(f"  = NEUTRAL (identical):    {neutral_count}/{total} "
          f"({100*neutral_count/max(total,1):.1f}%)")
    print(f"  ✗ UNSAFE (missed bug):    {unsafe_count}/{total} "
          f"({100*unsafe_count/max(total,1):.1f}%)")

    # Compute adjusted safety rate
    total_snippets = len(data['results'])
    exact_matches = sum(1 for r in data['results'] if r['exact_match'])
    safe_misses = safe_count + neutral_count  # not dangerous
    adjusted_safe = exact_matches + safe_misses
    print(f"\n  Adjusted safety analysis:")
    print(f"    Exact matches: {exact_matches}/{total_snippets}")
    print(f"    Safe near-misses: {safe_misses}/{total}")
    print(f"    Effective safe rate: {adjusted_safe}/{total_snippets} "
          f"({100*adjusted_safe/max(total_snippets,1):.1f}%)")
    if unsafe_count > 0:
        print(f"    ⚠ {unsafe_count} cases with potential missed portability bugs")
    else:
        print(f"    ✓ ZERO false negatives — all near-misses are safe or neutral")

    # Save results
    output = {
        'analysis': 'false_negative_classification',
        'total_near_misses': total,
        'safe_conservative': safe_count,
        'neutral_identical': neutral_count,
        'unsafe_missed': unsafe_count,
        'exact_matches': exact_matches,
        'total_snippets': total_snippets,
        'effective_safe_rate': round(100 * adjusted_safe / max(total_snippets, 1), 1),
        'zero_false_negatives': unsafe_count == 0,
        'results': results,
    }

    os.makedirs('paper_results_v4', exist_ok=True)
    with open('paper_results_v4/false_negative_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to paper_results_v4/false_negative_analysis.json")
    return output


if __name__ == '__main__':
    run_false_negative_analysis()
