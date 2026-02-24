#!/usr/bin/env python3
"""
Real-world package evaluation: run GuardHarvest on actual Python packages
and evaluate precision by manual inspection of reported bugs.

Downloads small, well-known packages and runs the analyzer.
"""

import ast
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.real_analyzer import FlowSensitiveAnalyzer


PACKAGES_TO_ANALYZE = [
    # (name, pip_name, source_subdir_glob)
    ("requests", "requests", "requests"),
    ("click", "click", "src/click"),
    ("flask", "flask", "src/flask"),
    ("httpx", "httpx", "httpx"),
    ("pydantic-core", "pydantic", "pydantic"),
    ("rich", "rich", "rich"),
    ("fastapi", "fastapi", "fastapi"),
]

DOWNLOAD_DIR = Path(__file__).parent / "real_packages"


def get_package_source(pip_name: str) -> Optional[Path]:
    """Find installed package source directory."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {pip_name}; print({pip_name}.__file__)"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            pkg_file = result.stdout.strip()
            return Path(pkg_file).parent
    except Exception:
        pass
    return None


def find_python_files(directory: Path, max_files: int = 50) -> List[Path]:
    """Find Python files in directory, excluding tests."""
    files = []
    for py_file in sorted(directory.rglob("*.py")):
        parts = py_file.parts
        if any(p in ("test", "tests", "__pycache__", ".git", "vendor") for p in parts):
            continue
        files.append(py_file)
        if len(files) >= max_files:
            break
    return files


def analyze_file(filepath: Path) -> Dict:
    """Analyze a single file and return results."""
    try:
        source = filepath.read_text(errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return {"file": str(filepath), "error": "parse_error", "bugs": [], "functions": 0, "guards": 0}

    analyzer = FlowSensitiveAnalyzer(source)
    all_bugs = []
    func_count = 0
    total_guards = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_count += 1
            try:
                result = analyzer.analyze_function(node)
                total_guards += result.guards_harvested
                for bug in result.bugs:
                    cat = bug.category if isinstance(bug.category, str) else (
                        bug.category.value if hasattr(bug.category, 'value') else str(bug.category)
                    )
                    all_bugs.append({
                        "category": cat,
                        "message": str(bug.message) if hasattr(bug, 'message') else str(bug),
                        "line": bug.line if hasattr(bug, 'line') else 0,
                        "function": node.name,
                    })
            except Exception:
                pass

    return {
        "file": str(filepath),
        "bugs": all_bugs,
        "functions": func_count,
        "guards": total_guards,
        "lines": len(source.splitlines()),
    }


def analyze_package_from_source(name: str, src_dir: Path) -> Dict:
    """Analyze a package source directory."""
    py_files = find_python_files(src_dir, max_files=100)
    if not py_files:
        return {"package": name, "error": "no Python files found", "files": []}

    total_bugs = []
    total_guards = 0
    total_functions = 0
    total_lines = 0
    file_results = []

    t0 = time.time()
    for py_file in py_files:
        result = analyze_file(py_file)
        file_results.append(result)
        total_bugs.extend(result.get("bugs", []))
        total_guards += result.get("guards", 0)
        total_functions += result.get("functions", 0)
        total_lines += result.get("lines", 0)
    elapsed = time.time() - t0

    # Categorize bugs
    by_category = {}
    for bug in total_bugs:
        cat = bug["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "package": name,
        "files_analyzed": len(py_files),
        "functions_analyzed": total_functions,
        "lines_analyzed": total_lines,
        "guards_harvested": total_guards,
        "guard_density_per_kloc": round(total_guards / max(1, total_lines / 1000), 1),
        "total_bugs_reported": len(total_bugs),
        "bug_density_per_kloc": round(len(total_bugs) / max(1, total_lines / 1000), 1),
        "bugs_by_category": by_category,
        "duration_seconds": round(elapsed, 2),
        "throughput_loc_per_sec": round(total_lines / max(0.001, elapsed)),
        "sample_bugs": total_bugs[:20],  # first 20 for inspection
    }


def run_on_stdlib():
    """Analyze Python standard library modules as ground truth."""
    import importlib
    stdlib_modules = ["json", "csv", "email", "http", "urllib", "xml", "collections",
                      "pathlib", "configparser", "argparse", "textwrap", "shutil"]
    results = []

    for mod_name in stdlib_modules:
        try:
            mod = importlib.import_module(mod_name)
            mod_file = getattr(mod, "__file__", None)
            if mod_file:
                mod_path = Path(mod_file)
                if mod_path.is_file():
                    result = analyze_file(mod_path)
                    result["module"] = mod_name
                    results.append(result)
                elif mod_path.parent.is_dir():
                    # Package with __init__.py
                    pkg_result = analyze_package_from_source(mod_name, mod_path.parent)
                    results.append(pkg_result)
        except Exception as e:
            results.append({"module": mod_name, "error": str(e)})

    return results


def main():
    print("=" * 60)
    print("GuardHarvest Real-World Package Evaluation")
    print("=" * 60)

    all_results = {}

    # 1. Run on stdlib
    print("\n--- Standard Library Analysis ---")
    stdlib_results = run_on_stdlib()
    all_results["stdlib"] = stdlib_results

    for r in stdlib_results:
        name = r.get("module", r.get("package", "?"))
        bugs = len(r.get("bugs", [])) if "bugs" in r else r.get("total_bugs_reported", 0)
        lines = r.get("lines", r.get("lines_analyzed", 0))
        guards = r.get("guards", r.get("guards_harvested", 0))
        print(f"  {name:20s}: {lines:5d} LOC, {guards:3d} guards, {bugs:3d} bugs reported")

    # 2. Run on installed packages
    print("\n--- Installed Package Analysis ---")
    pkg_results = []
    for name, pip_name, _ in PACKAGES_TO_ANALYZE:
        src_dir = get_package_source(pip_name)
        if src_dir and src_dir.exists():
            print(f"  Analyzing {name} from {src_dir}...")
            result = analyze_package_from_source(name, src_dir)
            pkg_results.append(result)
            print(f"    {result['lines_analyzed']} LOC, {result['guards_harvested']} guards, "
                  f"{result['total_bugs_reported']} bugs, "
                  f"{result['throughput_loc_per_sec']} LOC/sec")
        else:
            print(f"  {name}: not installed, skipping")
            pkg_results.append({"package": name, "error": "not installed"})

    all_results["packages"] = pkg_results

    # 3. Guard survey summary
    print("\n--- Guard Survey Summary ---")
    total_loc = 0
    total_guards = 0
    total_bugs = 0
    for r in stdlib_results + pkg_results:
        loc = r.get("lines", r.get("lines_analyzed", 0))
        g = r.get("guards", r.get("guards_harvested", 0))
        b = len(r.get("bugs", [])) if "bugs" in r else r.get("total_bugs_reported", 0)
        total_loc += loc
        total_guards += g
        total_bugs += b

    print(f"  Total LOC analyzed:   {total_loc}")
    print(f"  Total guards found:   {total_guards}")
    print(f"  Guard density:        {total_guards / max(1, total_loc / 1000):.1f} per KLOC")
    print(f"  Total bugs reported:  {total_bugs}")
    print(f"  Bug density:          {total_bugs / max(1, total_loc / 1000):.1f} per KLOC")

    all_results["guard_survey"] = {
        "total_loc": total_loc,
        "total_guards": total_guards,
        "guard_density_per_kloc": round(total_guards / max(1, total_loc / 1000), 1),
        "total_bugs_reported": total_bugs,
        "bug_density_per_kloc": round(total_bugs / max(1, total_loc / 1000), 1),
    }

    # Save results
    out_path = Path(__file__).parent / "results" / "realworld_package_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
