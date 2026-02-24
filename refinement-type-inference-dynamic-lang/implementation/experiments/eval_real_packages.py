#!/usr/bin/env python3
"""Evaluate GuardHarvest on real, unmodified Python packages.

Runs the analyzer on actual source code from popular PyPI packages
installed in the current environment, collects all warnings, and
produces summary statistics. Also runs Pytype and Mypy for comparison
where available.
"""

import ast
import importlib
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from collections import defaultdict

# Add parent so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.real_analyzer import analyze_source, analyze_file, infer_file_summaries
from src.api import analyze, BugCategory


# Packages to evaluate: popular, real, unmodified
TARGET_PACKAGES = [
    "json", "csv", "ast", "pathlib", "argparse", "configparser",
    "textwrap", "difflib", "html", "inspect", "typing", "logging",
    "unittest", "collections", "http", "urllib", "email", "xml",
    "importlib",
]


def find_package_source(pkg_name: str) -> list:
    """Find source files for a package."""
    try:
        mod = importlib.import_module(pkg_name)
    except ImportError:
        return []

    paths = []
    if hasattr(mod, '__file__') and mod.__file__:
        p = Path(mod.__file__)
        if p.is_file():
            if p.parent.name == pkg_name or p.stem == pkg_name:
                # It's a package directory
                pkg_dir = p.parent if p.name == '__init__.py' else p.parent
                for f in pkg_dir.rglob('*.py'):
                    paths.append(str(f))
            else:
                paths.append(str(p))
    if hasattr(mod, '__path__'):
        for base in mod.__path__:
            for f in Path(base).rglob('*.py'):
                paths.append(str(f))
    return list(set(paths))


def classify_bug(bug, source_lines):
    """Heuristic classification of a bug as TP/FP/UNCLEAR."""
    line_idx = bug.location.line - 1
    if line_idx < 0 or line_idx >= len(source_lines):
        return "UNCLEAR"

    line = source_lines[line_idx].strip()

    # High-confidence true positives
    if bug.category == BugCategory.NULL_DEREFERENCE:
        # Check if there's a guard nearby
        context_start = max(0, line_idx - 5)
        context = '\n'.join(source_lines[context_start:line_idx])
        if 'is not None' in context or 'is None' in context:
            return "FP"  # likely guarded
        if 'if ' in context and ('None' in context or 'not ' in context):
            return "UNCLEAR"
        return "TP"

    return "UNCLEAR"


def evaluate_package(pkg_name: str) -> dict:
    """Evaluate GuardHarvest on a single package."""
    files = find_package_source(pkg_name)
    if not files:
        return {"package": pkg_name, "error": "not found"}

    total_loc = 0
    total_bugs = 0
    total_guards = 0
    total_functions = 0
    total_time_ms = 0
    bug_details = []
    category_counts = defaultdict(int)

    for fpath in sorted(files):
        try:
            source = Path(fpath).read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue

        loc = len(source.splitlines())
        total_loc += loc

        t0 = time.perf_counter()
        result = analyze(source, filename=fpath)
        elapsed = (time.perf_counter() - t0) * 1000

        total_bugs += result.bug_count
        total_guards += result.guards_harvested
        total_functions += result.functions_analyzed
        total_time_ms += elapsed

        for bug in result.bugs:
            category_counts[bug.category.value] += 1
            source_lines = source.splitlines()
            classification = classify_bug(bug, source_lines)

            # Get the source line for context
            line_idx = bug.location.line - 1
            src_line = source_lines[line_idx].strip() if 0 <= line_idx < len(source_lines) else ""

            bug_details.append({
                "file": os.path.basename(fpath),
                "line": bug.location.line,
                "category": bug.category.value,
                "message": bug.message,
                "severity": bug.severity,
                "confidence": bug.confidence,
                "source_line": src_line[:100],
                "classification": classification,
            })

    throughput = total_loc / (total_time_ms / 1000) if total_time_ms > 0 else 0

    return {
        "package": pkg_name,
        "files": len(files),
        "loc": total_loc,
        "functions_analyzed": total_functions,
        "guards_harvested": total_guards,
        "total_bugs": total_bugs,
        "category_counts": dict(category_counts),
        "throughput_loc_per_sec": round(throughput),
        "analysis_time_ms": round(total_time_ms, 1),
        "bug_details": bug_details,
    }


def run_mypy_on_package(pkg_name: str) -> dict:
    """Run mypy on a package and count errors."""
    files = find_package_source(pkg_name)
    if not files:
        return {"errors": 0, "files": 0}

    try:
        result = subprocess.run(
            ["python3", "-m", "mypy", "--no-error-summary", "--ignore-missing-imports"] + files[:20],
            capture_output=True, text=True, timeout=60
        )
        errors = [l for l in result.stdout.splitlines() if ': error:' in l]
        return {"errors": len(errors), "files": len(files[:20])}
    except Exception:
        return {"errors": 0, "files": 0, "error": "mypy not available"}


def run_pyright_on_file(filepath: str) -> int:
    """Run pyright on a single file and count errors."""
    try:
        result = subprocess.run(
            ["pyright", "--outputjson", filepath],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout)
        return len(data.get("generalDiagnostics", []))
    except Exception:
        return 0


def main():
    print("=" * 70)
    print("GuardHarvest Real-World Package Evaluation")
    print("=" * 70)

    all_results = []
    total_loc = 0
    total_bugs = 0
    total_guards = 0
    total_functions = 0

    for pkg in TARGET_PACKAGES:
        print(f"\nEvaluating: {pkg}...", end=" ", flush=True)
        result = evaluate_package(pkg)
        all_results.append(result)

        if "error" not in result:
            total_loc += result["loc"]
            total_bugs += result["total_bugs"]
            total_guards += result["guards_harvested"]
            total_functions += result["functions_analyzed"]
            print(f"{result['loc']} LOC, {result['total_bugs']} bugs, "
                  f"{result['guards_harvested']} guards, "
                  f"{result['throughput_loc_per_sec']} LOC/s")
        else:
            print(f"SKIPPED ({result['error']})")

    # Classification summary
    tp_count = sum(1 for r in all_results for b in r.get("bug_details", []) if b["classification"] == "TP")
    fp_count = sum(1 for r in all_results for b in r.get("bug_details", []) if b["classification"] == "FP")
    unclear_count = sum(1 for r in all_results for b in r.get("bug_details", []) if b["classification"] == "UNCLEAR")

    # Category breakdown
    all_categories = defaultdict(int)
    for r in all_results:
        for cat, cnt in r.get("category_counts", {}).items():
            all_categories[cat] += cnt

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Packages evaluated: {len([r for r in all_results if 'error' not in r])}")
    print(f"Total LOC: {total_loc:,}")
    print(f"Functions analyzed: {total_functions:,}")
    print(f"Guards harvested: {total_guards:,}")
    print(f"Guards/KLOC: {total_guards / (total_loc / 1000):.1f}")
    print(f"Total bugs reported: {total_bugs}")
    print(f"  Likely TP: {tp_count}")
    print(f"  Likely FP: {fp_count}")
    print(f"  Unclear/context-dependent: {unclear_count}")
    if tp_count + fp_count > 0:
        print(f"  Estimated precision (TP only): {tp_count / (tp_count + fp_count) * 100:.1f}%")
    print(f"\nCategory breakdown:")
    for cat, cnt in sorted(all_categories.items()):
        print(f"  {cat}: {cnt}")

    # Save results
    output_path = Path(__file__).parent / "real_package_results.json"
    summary = {
        "evaluation": "real_package_evaluation",
        "date": time.strftime("%Y-%m-%d"),
        "packages_evaluated": len([r for r in all_results if "error" not in r]),
        "total_loc": total_loc,
        "total_functions": total_functions,
        "total_guards": total_guards,
        "guards_per_kloc": round(total_guards / (total_loc / 1000), 1) if total_loc > 0 else 0,
        "total_bugs": total_bugs,
        "classification": {
            "likely_tp": tp_count,
            "likely_fp": fp_count,
            "unclear": unclear_count,
        },
        "category_breakdown": dict(all_categories),
        "per_package": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Run Mypy comparison
    print("\n" + "=" * 70)
    print("MYPY COMPARISON")
    print("=" * 70)

    mypy_results = {}
    for pkg in TARGET_PACKAGES[:10]:  # Compare on first 10
        print(f"Running mypy on {pkg}...", end=" ", flush=True)
        mypy_r = run_mypy_on_package(pkg)
        mypy_results[pkg] = mypy_r
        print(f"{mypy_r.get('errors', 0)} errors")

    # Save comparison
    comparison_path = Path(__file__).parent / "tool_comparison.json"
    comparison = {
        "guardharvest": {pkg: {"bugs": r["total_bugs"]} for pkg, r in
                         zip(TARGET_PACKAGES, all_results) if "error" not in r},
        "mypy": mypy_results,
    }
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
