#!/usr/bin/env python3
"""
CI/CD Integration Demo — Run litmus-check on a sample concurrent C project.

This script demonstrates how LITMUS∞ integrates into a CI/CD pipeline.
It scans a sample C file with intentional portability issues, captures
the output, and saves results as JSON.

Usage:
    python3 run_demo.py
    # or:
    cd litmus_inf && litmus-check --target arm examples/ci_demo/concurrent_queue.c
"""

import json
import os
import sys
import time

# Ensure litmus_inf is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ast_analyzer import ASTAnalyzer
from litmus_check import find_source_files, extract_concurrency_snippets, check_snippet


def run_demo():
    """Run the full CI/CD demo and capture results."""
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(demo_dir, "concurrent_queue.c")

    if not os.path.exists(sample_file):
        print(f"ERROR: Sample file not found: {sample_file}")
        sys.exit(1)

    analyzer = ASTAnalyzer()
    target_arch = "arm"
    source_arch = "x86"

    print("=" * 60)
    print("LITMUS∞ CI/CD Integration Demo")
    print("=" * 60)
    print(f"\nTarget: {source_arch} → {target_arch}")
    print(f"File:   {os.path.basename(sample_file)}")
    print()

    # Time the analysis
    start = time.perf_counter()

    # Extract concurrency snippets
    snippets = extract_concurrency_snippets(sample_file)
    all_results = []
    coverage_data = []

    for snip in snippets:
        results, cov = check_snippet(
            analyzer, snip['code'], target_arch, source_arch,
            warn_unrecognized=True
        )
        for r in results:
            r['file'] = os.path.basename(snip['file'])
            r['start_line'] = snip['start_line']
            all_results.append(r)
        if cov:
            coverage_data.append({
                'start_line': snip['start_line'],
                'coverage_confidence': cov['coverage_confidence'],
                'warnings': cov['warnings'],
            })

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Print results
    unsafe = [r for r in all_results if not r['safe']]
    safe = [r for r in all_results if r['safe']]

    if unsafe:
        print(f"\033[31m✗ {len(unsafe)} portability issue(s) found "
              f"({source_arch} → {target_arch}):\033[0m\n")
        for r in unsafe:
            print(f"  {r['file']}:{r['start_line']}: "
                  f"\033[31m✗ UNSAFE\033[0m  {r['pattern']} → {r['target_arch']}  "
                  f"(confidence: {r.get('confidence', 0):.0%})")
            if r.get('fence_fix'):
                print(f"    Fix: {r['fence_fix']}")
            print()

    if safe:
        print(f"\033[32m✓ {len(safe)} safe pattern(s):\033[0m")
        for r in safe:
            print(f"  {r['file']}:{r['start_line']}: "
                  f"\033[32m✓ SAFE\033[0m  {r['pattern']} → {r['target_arch']}  "
                  f"(confidence: {r.get('confidence', 0):.0%})")
        print()

    # Coverage warnings
    has_warnings = any(c['warnings'] for c in coverage_data)
    if has_warnings:
        print("\033[33m⚠ Coverage warnings:\033[0m")
        for c in coverage_data:
            for w in c['warnings']:
                print(f"  Line {c['start_line']}: {w}")
        print()

    print(f"Summary: {len(all_results)} pattern(s) checked, "
          f"{len(unsafe)} issue(s) in {elapsed_ms:.0f}ms")

    # Explain findings
    print("\n" + "=" * 60)
    print("What Each Finding Means")
    print("=" * 60)

    explanations = {
        "mp": (
            "Message Passing (MP): Thread 0 writes data then flag; "
            "Thread 1 reads flag then data. On ARM, the flag store can "
            "be reordered before the data store, so Thread 1 may observe "
            "flag=1 but data=0. Fix: add DMB fences."
        ),
        "sb": (
            "Store Buffering (SB): Both threads write to their own "
            "variable then read the other. Store buffer forwarding "
            "can cause both reads to see stale values."
        ),
        "lb": (
            "Load Buffering (LB): Each thread reads then writes. On "
            "ARM/RISC-V, speculative execution can produce 'out of thin "
            "air' values. This is unsafe on all weakly ordered architectures."
        ),
    }

    seen_pats = set()
    for r in unsafe:
        pat = r['pattern']
        base_pat = pat.split('_')[0] if '_' in pat else pat
        if base_pat not in seen_pats and base_pat in explanations:
            seen_pats.add(base_pat)
            print(f"\n  {pat}: {explanations[base_pat]}")

    # Save JSON output
    output = {
        "demo": "CI/CD Integration Demo",
        "source_file": os.path.basename(sample_file),
        "source_arch": source_arch,
        "target_arch": target_arch,
        "elapsed_ms": round(elapsed_ms, 1),
        "total_patterns_checked": len(all_results),
        "unsafe_count": len(unsafe),
        "safe_count": len(safe),
        "results": all_results,
        "coverage": coverage_data,
        "ci_exit_code": 1 if unsafe else 0,
        "github_action_config": {
            "workflow_file": ".github/workflows/litmus-check.yml",
            "trigger": "pull_request (on C/C++/CUDA file changes)",
            "command": "litmus-check --target arm --json src/",
            "artifacts": "litmus-results.json",
        }
    }

    out_path = os.path.join(demo_dir, "demo_output.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    output = run_demo()
    sys.exit(output["ci_exit_code"])
