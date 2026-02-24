#!/usr/bin/env python3
"""
Real-world evaluation of the flow-sensitive refinement type analyzer.

Analyzes real Python packages from the local stdlib and site-packages
to evaluate guard-harvesting, scalability, and bug-detection accuracy.
"""

import sys
import os
import ast
import json
import time
import sysconfig
import site
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.real_analyzer import analyze_file, analyze_directory, analyze_source, FlowSensitiveAnalyzer


# ── Configuration ───────────────────────────────────────────────────

STDLIB_PACKAGES = [
    "json", "email", "http", "unittest", "xml", "logging",
    "configparser", "argparse", "csv", "urllib", "html",
    "collections", "importlib", "pathlib", "typing",
    "ast", "dis", "inspect", "textwrap", "difflib",
]

MIN_TOTAL_LOC = 10_000
MIN_PACKAGES = 10

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────

def find_package_files(base_dir: str, package_name: str) -> list[Path]:
    """Find all .py files for a package under base_dir."""
    base = Path(base_dir)
    # Package could be a directory or a single .py file
    pkg_dir = base / package_name
    pkg_file = base / f"{package_name}.py"
    files = []
    if pkg_dir.is_dir():
        for f in sorted(pkg_dir.rglob("*.py")):
            if "__pycache__" not in str(f):
                files.append(f)
    elif pkg_file.is_file():
        files.append(pkg_file)
    return files


def count_loc(files: list[Path]) -> int:
    total = 0
    for f in files:
        try:
            total += len(f.read_text(encoding="utf-8", errors="replace").splitlines())
        except Exception:
            pass
    return total


def count_guards_in_source(source: str) -> dict:
    """Count guard patterns in source code via AST."""
    counts = Counter()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return counts

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            _count_test_guards(node.test, counts)
        elif isinstance(node, ast.Assert):
            _count_test_guards(node.test, counts)
        # Ternary (IfExp)
        elif isinstance(node, ast.IfExp):
            _count_test_guards(node.test, counts)
    return counts


def _count_test_guards(test, counts):
    """Classify a test expression into guard categories."""
    if isinstance(test, ast.Call):
        if isinstance(test.func, ast.Name) and test.func.id == "isinstance":
            counts["isinstance"] += 1
        elif isinstance(test.func, ast.Name) and test.func.id == "callable":
            counts["callable_check"] += 1
        elif isinstance(test.func, ast.Name) and test.func.id == "hasattr":
            counts["hasattr_check"] += 1
        elif isinstance(test.func, ast.Name) and test.func.id in ("issubclass",):
            counts["issubclass"] += 1
        else:
            counts["other_call"] += 1
    elif isinstance(test, ast.Compare) and len(test.ops) >= 1:
        op = test.ops[0]
        right = test.comparators[0] if test.comparators else None
        if isinstance(op, (ast.Is, ast.IsNot)):
            if right and isinstance(right, ast.Constant) and right.value is None:
                counts["is_none / is_not_none"] += 1
            else:
                counts["is_identity"] += 1
        elif isinstance(op, (ast.Eq, ast.NotEq)):
            if right and isinstance(right, ast.Constant) and right.value is None:
                counts["eq_none"] += 1
            else:
                counts["equality_comparison"] += 1
        elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            counts["numeric_comparison"] += 1
        elif isinstance(op, (ast.In, ast.NotIn)):
            counts["membership_test"] += 1
        else:
            counts["other_comparison"] += 1
    elif isinstance(test, ast.BoolOp):
        for val in test.values:
            _count_test_guards(val, counts)
    elif isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        _count_test_guards(test.operand, counts)
        counts["negation"] += 1
    elif isinstance(test, ast.Name):
        counts["truthiness_check"] += 1
    elif isinstance(test, ast.Attribute):
        counts["attribute_truthiness"] += 1
    else:
        counts["other_guard"] += 1


def get_source_context(filepath: str, line: int, context: int = 2) -> list[str]:
    """Return source lines around the given line number."""
    try:
        lines = Path(filepath).read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, line - 1 - context)
        end = min(len(lines), line + context)
        result = []
        for i in range(start, end):
            marker = ">>>" if i == line - 1 else "   "
            result.append(f"{marker} {i+1:4d} | {lines[i]}")
        return result
    except Exception:
        return [f"   (could not read {filepath})"]


# ── Main experiment runner ──────────────────────────────────────────

def run_realworld_experiments() -> dict:
    stdlib = sysconfig.get_path("stdlib")
    site_packages_dirs = []
    try:
        site_packages_dirs = site.getsitepackages()
    except Exception:
        pass
    # Also try user site
    try:
        usp = site.getusersitepackages()
        if usp and Path(usp).is_dir():
            site_packages_dirs.append(usp)
    except Exception:
        pass

    print(f"stdlib: {stdlib}")
    print(f"site-packages: {site_packages_dirs}")

    # ── 1. Discover packages ────────────────────────────────────────
    print("\n=== Phase 1: Discovering packages ===")
    packages = {}  # name -> list[Path]

    # Stdlib packages
    for pkg in STDLIB_PACKAGES:
        files = find_package_files(stdlib, pkg)
        if files:
            packages[f"stdlib/{pkg}"] = files

    # Site-packages: pick targeted well-known packages
    SITE_TARGETS = [
        "pip", "setuptools", "pkg_resources", "certifi",
        "charset_normalizer", "idna", "requests", "six",
        "dateutil", "yaml", "toml", "tomli", "packaging",
    ]
    for sp_dir in site_packages_dirs:
        if not Path(sp_dir).is_dir():
            continue
        for pkg_name in SITE_TARGETS:
            if f"site/{pkg_name}" in packages:
                continue
            files = find_package_files(sp_dir, pkg_name)
            if files:
                # Limit to 200 files max per package to stay fast
                files = files[:200]
                packages[f"site/{pkg_name}"] = files

    # Report
    total_loc = 0
    total_files = 0
    for name, files in sorted(packages.items()):
        loc = count_loc(files)
        total_loc += loc
        total_files += len(files)
        print(f"  {name}: {len(files)} files, {loc} LOC")

    print(f"\nTotal: {len(packages)} packages, {total_files} files, {total_loc} LOC")

    if len(packages) < MIN_PACKAGES:
        print(f"WARNING: Only found {len(packages)} packages (need {MIN_PACKAGES})")
    if total_loc < MIN_TOTAL_LOC:
        print(f"WARNING: Only found {total_loc} LOC (need {MIN_TOTAL_LOC})")

    # ── 2. Guard survey ─────────────────────────────────────────────
    print("\n=== Phase 2: Guard survey ===")
    guard_survey = {}
    total_guard_counts = Counter()

    for pkg_name, files in sorted(packages.items()):
        pkg_guards = Counter()
        for f in files:
            try:
                source = f.read_text(encoding="utf-8", errors="replace")
                gc = count_guards_in_source(source)
                pkg_guards += gc
            except Exception:
                pass
        guard_survey[pkg_name] = dict(pkg_guards)
        total_guard_counts += pkg_guards
        total_g = sum(pkg_guards.values())
        if total_g > 0:
            top3 = pkg_guards.most_common(3)
            print(f"  {pkg_name}: {total_g} guards — top: {top3}")

    guard_survey["_total"] = dict(total_guard_counts)
    total_all_guards = sum(total_guard_counts.values())
    print(f"\nGuard distribution ({total_all_guards} total):")
    for gtype, cnt in total_guard_counts.most_common():
        pct = cnt / total_all_guards * 100 if total_all_guards else 0
        print(f"  {gtype}: {cnt} ({pct:.1f}%)")

    # ── 3. Analyze with guards (full analysis) ──────────────────────
    print("\n=== Phase 3: Full analysis (guards enabled) ===")
    all_results_with_guards = {}
    all_bugs_with_guards = []
    scalability_data = []

    for pkg_name, files in sorted(packages.items()):
        pkg_start = time.perf_counter()
        pkg_loc = 0
        pkg_funcs = 0
        pkg_guards_found = 0
        pkg_bugs = 0
        pkg_files_ok = 0

        for f in files:
            try:
                result = analyze_file(str(f), use_cegar=False)
                pkg_loc += result.lines_of_code
                pkg_funcs += result.functions_analyzed
                pkg_guards_found += result.total_guards
                pkg_bugs += result.total_bugs
                pkg_files_ok += 1

                # Collect bugs with file info
                for fr in result.function_results:
                    for bug in fr.bugs:
                        all_bugs_with_guards.append({
                            "package": pkg_name,
                            "file": str(f),
                            "function": bug.function,
                            "line": bug.line,
                            "col": bug.col,
                            "category": bug.category.name,
                            "message": bug.message,
                            "variable": bug.variable,
                            "severity": bug.severity,
                        })
            except Exception as e:
                pass

        pkg_elapsed = (time.perf_counter() - pkg_start) * 1000
        entry = {
            "package": pkg_name,
            "files": pkg_files_ok,
            "loc": pkg_loc,
            "functions": pkg_funcs,
            "guards": pkg_guards_found,
            "bugs": pkg_bugs,
            "time_ms": round(pkg_elapsed, 1),
        }
        all_results_with_guards[pkg_name] = entry
        scalability_data.append(entry)
        print(f"  {pkg_name}: {pkg_loc} LOC, {pkg_funcs} funcs, "
              f"{pkg_guards_found} guards, {pkg_bugs} bugs, {pkg_elapsed:.0f}ms")

    # ── 4. Analyze without guards (baseline) ────────────────────────
    print("\n=== Phase 4: Baseline analysis (guards disabled / no narrowing) ===")
    all_results_no_guards = {}
    all_bugs_no_guards = []

    for pkg_name, files in sorted(packages.items()):
        pkg_bugs = 0
        pkg_funcs = 0
        pkg_loc = 0

        for f in files:
            try:
                source = f.read_text(encoding="utf-8", errors="replace")
                try:
                    tree = ast.parse(source, str(f))
                except SyntaxError:
                    continue

                analyzer = FlowSensitiveAnalyzer(source, str(f))
                func_nodes = [n for n in ast.walk(tree)
                              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

                # Disable guard narrowing: _apply_guards returns env unchanged
                orig_apply = analyzer._apply_guards
                analyzer._apply_guards = lambda env, *a, **kw: env
                try:
                    for func_node in func_nodes:
                        try:
                            fr = analyzer.analyze_function(func_node)
                        except Exception:
                            continue
                        pkg_bugs += len(fr.bugs)
                        pkg_funcs += 1
                finally:
                    analyzer._apply_guards = orig_apply

                pkg_loc += len(source.splitlines())
            except Exception:
                pass

        all_results_no_guards[pkg_name] = {
            "package": pkg_name,
            "loc": pkg_loc,
            "functions": pkg_funcs,
            "bugs_no_guards": pkg_bugs,
        }
        bugs_with = all_results_with_guards.get(pkg_name, {}).get("bugs", 0)
        print(f"  {pkg_name}: with_guards={bugs_with}, no_guards={pkg_bugs}")

    # ── 5. Verified sample ──────────────────────────────────────────
    print("\n=== Phase 5: Manual verification sample (first 20 bugs) ===")
    verified_sample = []
    for bug_info in all_bugs_with_guards[:20]:
        context_lines = get_source_context(bug_info["file"], bug_info["line"], context=2)
        entry = {
            **bug_info,
            "source_context": context_lines,
        }
        verified_sample.append(entry)
        print(f"\n  [{bug_info['category']}] {bug_info['file']}:{bug_info['line']} "
              f"in {bug_info['function']}")
        print(f"    {bug_info['message']}")
        for cl in context_lines:
            print(f"    {cl}")

    # ── 6. Compile results ──────────────────────────────────────────
    print("\n=== Phase 6: Saving results ===")

    # Comparison summary
    comparison = {}
    for pkg_name in all_results_with_guards:
        wg = all_results_with_guards[pkg_name]
        ng = all_results_no_guards.get(pkg_name, {})
        bugs_with = wg.get("bugs", 0)
        bugs_without = ng.get("bugs_no_guards", 0)
        reduction = bugs_without - bugs_with
        pct = (reduction / bugs_without * 100) if bugs_without > 0 else 0
        comparison[pkg_name] = {
            "bugs_with_guards": bugs_with,
            "bugs_without_guards": bugs_without,
            "bugs_eliminated_by_guards": reduction,
            "reduction_pct": round(pct, 1),
        }

    total_with = sum(v["bugs_with_guards"] for v in comparison.values())
    total_without = sum(v["bugs_without_guards"] for v in comparison.values())
    total_reduction = total_without - total_with
    total_pct = (total_reduction / total_without * 100) if total_without > 0 else 0

    summary = {
        "packages_analyzed": len(packages),
        "total_files": total_files,
        "total_loc": total_loc,
        "total_guards_surveyed": total_all_guards,
        "total_bugs_with_guards": total_with,
        "total_bugs_without_guards": total_without,
        "bugs_eliminated_by_guards": total_reduction,
        "guard_reduction_pct": round(total_pct, 1),
        "packages": list(all_results_with_guards.keys()),
    }

    # Sort scalability by LOC
    scalability_data.sort(key=lambda x: x["loc"])
    scalability = {
        "by_package": scalability_data,
        "throughput_loc_per_sec": [],
    }
    for entry in scalability_data:
        if entry["time_ms"] > 0:
            throughput = entry["loc"] / (entry["time_ms"] / 1000)
            scalability["throughput_loc_per_sec"].append({
                "package": entry["package"],
                "loc": entry["loc"],
                "time_ms": entry["time_ms"],
                "loc_per_sec": round(throughput, 0),
            })

    # Save files
    def save_json(data, filename):
        path = RESULTS_DIR / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved {path}")

    save_json(summary, "RW_realworld_summary.json")
    save_json(guard_survey, "RW_guard_survey.json")
    save_json(scalability, "RW_scalability.json")
    save_json(verified_sample, "RW_verified_bugs.json")
    save_json(comparison, "RW_comparison.json")

    all_results = {
        "summary": summary,
        "guard_survey": guard_survey,
        "scalability": scalability,
        "verified_sample": verified_sample,
        "comparison": comparison,
    }

    print("\n=== Done ===")
    print(f"Packages: {summary['packages_analyzed']}")
    print(f"Total LOC: {summary['total_loc']}")
    print(f"Bugs (with guards): {summary['total_bugs_with_guards']}")
    print(f"Bugs (no guards): {summary['total_bugs_without_guards']}")
    print(f"Guard reduction: {summary['guard_reduction_pct']}%")

    return all_results


if __name__ == "__main__":
    results = run_realworld_experiments()
