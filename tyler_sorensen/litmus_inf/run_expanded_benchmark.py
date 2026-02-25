#!/usr/bin/env python3
"""
Run expanded benchmark suite and produce results for paper.

Combines original 203 snippets + 194 new snippets = 397 total.
Runs AST analyzer on all, produces per-category accuracy breakdown.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_suite import BENCHMARK_SNIPPETS, run_benchmark
from expanded_benchmark_v3 import EXPANDED_BENCHMARK_SNIPPETS
from ast_analyzer import ast_analyze_code
from statistical_analysis import wilson_ci


def run_expanded_benchmark(output_dir='paper_results_v7'):
    """Run full expanded benchmark and save results."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_snippets = BENCHMARK_SNIPPETS + EXPANDED_BENCHMARK_SNIPPETS
    total = len(all_snippets)
    
    print("=" * 70)
    print(f"LITMUS∞ Expanded Benchmark Suite — {total} Code Snippets")
    print("=" * 70)
    
    # Run analyzer on all snippets
    results, summary = run_benchmark(ast_analyze_code, snippets=all_snippets)
    
    exact = summary['exact_correct']
    top3 = summary['top3_correct']
    exact_acc = summary['exact_accuracy']
    top3_acc = summary['top3_accuracy']
    
    print(f"\n{'='*60}")
    print(f"Overall Results:")
    print(f"  Total snippets: {total}")
    print(f"  Exact match: {exact}/{total} ({exact_acc:.1%})")
    print(f"  Top-3 match: {top3}/{total} ({top3_acc:.1%})")
    
    # Wilson CIs
    exact_ci = wilson_ci(exact, total)
    top3_ci = wilson_ci(top3, total)
    print(f"  Exact Wilson 95% CI: [{exact_ci[1]:.1%}, {exact_ci[2]:.1%}]")
    print(f"  Top-3 Wilson 95% CI: [{top3_ci[1]:.1%}, {top3_ci[2]:.1%}]")
    
    # Per-category breakdown
    print(f"\nPer-category accuracy:")
    categories = sorted(summary['per_category'].items(),
                       key=lambda x: x[1]['total'], reverse=True)
    for cat, stats in categories:
        print(f"  {cat:25s}: exact={stats['exact']}/{stats['total']} "
              f"({stats['exact_accuracy']:.0%}), "
              f"top3={stats['top3']}/{stats['total']} "
              f"({stats['top3_accuracy']:.0%})")
    
    # Per-project source breakdown
    projects = {}
    for r in results:
        cat = r.get('category', 'unknown')
        if cat not in projects:
            projects[cat] = {'total': 0, 'exact': 0, 'top3': 0}
        projects[cat]['total'] += 1
        if r.get('exact_match'):
            projects[cat]['exact'] += 1
        if r.get('top3_match'):
            projects[cat]['top3'] += 1
    
    # Show failures
    failures = [r for r in results if not r.get('top3_match')]
    print(f"\nFailures (not in top-3): {len(failures)}")
    for r in failures[:20]:  # Show first 20
        print(f"  ✗ {r['id']}: expected={r['expected']}, got={r.get('top3', [])}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    
    # Timing
    times = [r.get('time_ms', 0) for r in results if r.get('time_ms')]
    if times:
        avg_ms = sum(times) / len(times)
        max_ms = max(times)
        print(f"\nTiming: mean={avg_ms:.1f}ms, max={max_ms:.1f}ms")
    
    # Save results
    output = {
        'method': 'AST-based pattern recognition with expanded benchmark',
        'total_snippets': total,
        'original_snippets': len(BENCHMARK_SNIPPETS),
        'new_snippets': len(EXPANDED_BENCHMARK_SNIPPETS),
        'exact_correct': exact,
        'top3_correct': top3,
        'exact_accuracy': round(exact_acc, 4),
        'top3_accuracy': round(top3_acc, 4),
        'exact_wilson_ci': [round(exact_ci[1], 4), round(exact_ci[2], 4)],
        'top3_wilson_ci': [round(top3_ci[1], 4), round(top3_ci[2], 4)],
        'n_categories': len(summary['per_category']),
        'n_source_projects': len(set(
            s.get('category', '') for s in all_snippets
        )),
        'per_category': summary['per_category'],
        'failures': [{
            'id': r['id'],
            'expected': r['expected'],
            'predicted': r.get('predicted', 'none'),
            'top3': r.get('top3', []),
            'category': r.get('category', ''),
        } for r in failures],
        'timing': {
            'mean_ms': round(avg_ms, 2) if times else 0,
            'max_ms': round(max_ms, 2) if times else 0,
        },
        'results': results,
    }
    
    out_path = os.path.join(output_dir, 'expanded_benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_path}")
    return output


if __name__ == '__main__':
    run_expanded_benchmark()
