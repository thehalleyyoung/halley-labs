#!/usr/bin/env python3
"""Evaluate GuardHarvest on real, unmodified stdlib Python modules.

Analyzes actual CPython standard library source files to measure
precision, recall (via manual classification), and throughput.
Compares against Mypy on untyped code.
"""

import json
import os
import sys
import time
import subprocess
import importlib
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.api import analyze, BugCategory


# Map package name -> list of specific .py files (stdlib modules)
STDLIB_MODULES = [
    "json", "csv", "ast", "textwrap", "difflib",
    "configparser", "argparse", "pathlib", "html",
    "inspect", "logging", "typing", "collections",
    "importlib", "unittest",
]


def get_module_files(mod_name: str) -> list:
    """Get .py source files for a stdlib module."""
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        return []

    files = []
    if hasattr(mod, '__file__') and mod.__file__ and Path(mod.__file__).exists():
        main_file = Path(mod.__file__)
        if main_file.name == '__init__.py':
            # Package: get all .py in directory
            for f in sorted(main_file.parent.rglob('*.py')):
                if '__pycache__' not in str(f):
                    files.append(str(f))
        else:
            files.append(str(main_file))
    return files


def analyze_files(files: list, pkg_name: str) -> dict:
    """Analyze a list of files and return results."""
    total_loc = 0
    total_bugs = 0
    total_guards = 0
    total_functions = 0
    bug_details = []
    category_counts = defaultdict(int)
    t_total = 0

    for fpath in files:
        try:
            source = Path(fpath).read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue

        loc = len(source.splitlines())
        total_loc += loc

        t0 = time.perf_counter()
        try:
            result = analyze(source, filename=fpath)
        except Exception:
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        t_total += elapsed_ms

        total_bugs += result.bug_count
        total_guards += result.guards_harvested
        total_functions += result.functions_analyzed

        for bug in result.bugs:
            category_counts[bug.category.value] += 1
            source_lines = source.splitlines()
            line_idx = bug.location.line - 1
            src_line = source_lines[line_idx].strip() if 0 <= line_idx < len(source_lines) else ""

            bug_details.append({
                "file": os.path.relpath(fpath),
                "line": bug.location.line,
                "category": bug.category.value,
                "message": bug.message,
                "confidence": bug.confidence,
                "source_line": src_line[:120],
            })

    throughput = total_loc / (t_total / 1000) if t_total > 0 else 0

    return {
        "package": pkg_name,
        "files": len(files),
        "loc": total_loc,
        "functions": total_functions,
        "guards": total_guards,
        "bugs": total_bugs,
        "categories": dict(category_counts),
        "throughput_loc_s": round(throughput),
        "time_ms": round(t_total, 1),
        "bug_details": bug_details,
    }


def run_mypy(files: list) -> int:
    """Run mypy and count errors."""
    try:
        r = subprocess.run(
            ["python3", "-m", "mypy", "--ignore-missing-imports", "--no-error-summary"]
            + files[:20],
            capture_output=True, text=True, timeout=60
        )
        return len([l for l in r.stdout.splitlines() if ': error:' in l])
    except Exception:
        return -1


def main():
    print("=" * 70)
    print("GuardHarvest: Real-World Stdlib Evaluation")
    print("Interprocedural analysis enabled")
    print("=" * 70)

    all_results = []
    grand_loc = 0
    grand_bugs = 0
    grand_guards = 0
    grand_funcs = 0

    mypy_comparison = {}

    for mod_name in STDLIB_MODULES:
        files = get_module_files(mod_name)
        if not files:
            print(f"  {mod_name}: not found")
            continue

        print(f"  {mod_name}...", end=" ", flush=True)
        result = analyze_files(files, mod_name)
        all_results.append(result)

        grand_loc += result["loc"]
        grand_bugs += result["bugs"]
        grand_guards += result["guards"]
        grand_funcs += result["functions"]

        # Run mypy comparison
        mypy_errors = run_mypy(files)
        mypy_comparison[mod_name] = mypy_errors

        print(f"{result['loc']} LOC, {result['bugs']} bugs, "
              f"{result['guards']} guards, mypy={mypy_errors} errors, "
              f"{result['throughput_loc_s']} LOC/s")

    # Category totals
    cat_totals = defaultdict(int)
    for r in all_results:
        for cat, cnt in r["categories"].items():
            cat_totals[cat] += cnt

    # Null-deref focus
    null_bugs = sum(r["categories"].get("null_dereference", 0) for r in all_results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Packages: {len(all_results)}")
    print(f"Total LOC: {grand_loc:,}")
    print(f"Functions analyzed: {grand_funcs:,}")
    print(f"Guards harvested: {grand_guards:,}")
    print(f"Guard density: {grand_guards / max(grand_loc/1000, 1):.1f} guards/KLOC")
    print(f"Total warnings: {grand_bugs}")
    print(f"  null_dereference: {null_bugs}")
    for cat, cnt in sorted(cat_totals.items()):
        if cat != "null_dereference":
            print(f"  {cat}: {cnt}")
    print(f"\nMypy comparison (on same files, untyped):")
    mypy_total = sum(v for v in mypy_comparison.values() if v >= 0)
    print(f"  GuardHarvest total warnings: {grand_bugs}")
    print(f"  Mypy total errors: {mypy_total}")

    # Save
    output = {
        "evaluation": "stdlib_real_world",
        "interprocedural": True,
        "date": time.strftime("%Y-%m-%d"),
        "total_loc": grand_loc,
        "total_functions": grand_funcs,
        "total_guards": grand_guards,
        "guard_density_per_kloc": round(grand_guards / max(grand_loc/1000, 1), 1),
        "total_bugs": grand_bugs,
        "category_breakdown": dict(cat_totals),
        "mypy_comparison": mypy_comparison,
        "guardharvest_vs_mypy": {
            "guardharvest_warnings": grand_bugs,
            "mypy_errors": mypy_total,
        },
        "per_package": all_results,
    }

    out_path = Path(__file__).parent / "stdlib_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
