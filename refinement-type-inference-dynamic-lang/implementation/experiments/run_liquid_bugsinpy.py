#!/usr/bin/env python3
"""
BugsInPy evaluation comparing baseline FlowSensitiveAnalyzer vs LiquidTypeInferencer.

Reads bug patches from BugsInPy, extracts buggy (pre-fix) functions from diffs,
runs both analyzers, and compares recall.
"""

import ast
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.real_analyzer import FlowSensitiveAnalyzer
from src.liquid import analyze_liquid

BUGSINPY_ROOT = Path(__file__).parent / "BugsInPy" / "projects"
RESULTS_DIR = Path(__file__).parent / "results"

GH_DETECTABLE = {"null_dereference", "type_error", "division_by_zero",
                  "index_out_of_bounds", "attribute_error"}

LIQUID_TIMEOUT = 5  # seconds per function


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Analysis timed out")


def classify_bug_from_patch(patch_text: str) -> str:
    """Heuristically classify a bug from its patch content."""
    p = patch_text.lower()
    if any(k in p for k in ["is not none", "is none", "none", "nonetype",
                             "attributeerror", "has no attribute"]):
        return "null_dereference"
    if any(k in p for k in ["isinstance", "typeerror", "type(", "type error"]):
        return "type_error"
    if any(k in p for k in ["/ 0", "divisionerror", "zerodivision",
                             "division by zero", "!= 0", "> 0"]):
        return "division_by_zero"
    if any(k in p for k in ["indexerror", "index out of", "out of range",
                             "keyerror", "len(", "bounds"]):
        return "index_out_of_bounds"
    if any(k in p for k in ["getattr", "hasattr", "attribute"]):
        return "attribute_error"
    return "logic_or_other"


def extract_buggy_functions_from_patch(patch_text: str) -> List[Tuple[str, str]]:
    """Extract (function_name, buggy_code) pairs from unified diff."""
    functions = []
    current_func = None
    func_lines = []
    in_func = False

    for line in patch_text.split("\n"):
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            continue

        if line.startswith("@@"):
            m = re.search(r'def\s+(\w+)\s*\(', line)
            if m:
                if in_func and func_lines and current_func:
                    code = "\n".join(func_lines)
                    if code.strip():
                        functions.append((current_func, code))
                current_func = m.group(1)
                func_lines = []
                in_func = True
            continue

        if not in_func:
            continue

        # Skip added lines (fix code, not buggy code)
        if line.startswith("+"):
            continue

        if line.startswith("-"):
            actual_line = line[1:]
        elif line.startswith(" "):
            actual_line = line[1:]
        else:
            actual_line = line

        func_lines.append(actual_line)

    if in_func and func_lines and current_func:
        code = "\n".join(func_lines)
        if code.strip():
            functions.append((current_func, code))

    return functions


def make_analyzable_function(func_name: str, func_body: str) -> Optional[str]:
    """Try to make a function body parseable."""
    if func_body.strip().startswith("def "):
        try:
            ast.parse(func_body)
            return func_body
        except SyntaxError:
            pass

    indented = "\n".join("    " + l for l in func_body.split("\n"))
    wrapped = f"def {func_name}(*args, **kwargs):\n{indented}"
    try:
        ast.parse(wrapped)
        return wrapped
    except SyntaxError:
        pass

    lines = func_body.split("\n")
    for end in range(len(lines), max(0, len(lines) - 10), -1):
        subset = "\n".join("    " + l for l in lines[:end])
        code = f"def {func_name}(*args, **kwargs):\n{subset}"
        try:
            ast.parse(code)
            return code
        except SyntaxError:
            continue

    return None


def run_baseline_on_code(source: str) -> List[Dict]:
    """Run FlowSensitiveAnalyzer (baseline) on source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    analyzer = FlowSensitiveAnalyzer(source)
    all_bugs = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                result = analyzer.analyze_function(node)
                for bug in result.bugs:
                    cat = bug.category if isinstance(bug.category, str) else (
                        bug.category.value if hasattr(bug.category, 'value') else str(bug.category))
                    all_bugs.append({
                        "category": cat,
                        "message": str(getattr(bug, 'message', str(bug))),
                        "line": getattr(bug, 'line', 0),
                    })
            except Exception:
                pass

    return all_bugs


def run_liquid_on_code(source: str) -> List[Dict]:
    """Run LiquidTypeInferencer on source code with timeout."""
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(LIQUID_TIMEOUT)
        result = analyze_liquid(source)
        signal.alarm(0)

        bugs = []
        for bug in result.bugs:
            kind = bug.kind
            kind_str = kind.name.lower() if hasattr(kind, 'name') else str(kind)
            bugs.append({
                "category": kind_str,
                "message": bug.message,
                "line": bug.line,
                "variable": getattr(bug, 'variable', ''),
            })
        return bugs
    except TimeoutError:
        return []
    except Exception:
        return []
    finally:
        signal.alarm(0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)


def run_experiment():
    """Main experiment: compare baseline vs liquid on BugsInPy."""
    if not BUGSINPY_ROOT.exists():
        print(f"ERROR: BugsInPy root not found at {BUGSINPY_ROOT}")
        sys.exit(1)

    projects = sorted(p.name for p in BUGSINPY_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(projects)} projects in BugsInPy")

    total_bugs = 0
    bugs_in_scope = 0
    baseline_detected = 0
    liquid_detected = 0
    by_project = {}
    by_category = {}
    detailed_results = []

    for project in projects:
        bugs_dir = BUGSINPY_ROOT / project / "bugs"
        if not bugs_dir.exists():
            continue

        bug_dirs = sorted(
            (d for d in bugs_dir.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name)
        )

        proj = {"total": 0, "in_scope": 0, "baseline_detected": 0, "liquid_detected": 0}

        for bug_dir in bug_dirs:
            bug_id = bug_dir.name
            patch_file = bug_dir / "bug_patch.txt"
            total_bugs += 1
            proj["total"] += 1

            entry = {
                "project": project,
                "bug_id": bug_id,
                "category": "unknown",
                "in_scope": False,
                "baseline_detected": False,
                "liquid_detected": False,
                "baseline_bugs": [],
                "liquid_bugs": [],
                "functions_extracted": 0,
                "error": None,
            }

            try:
                if not patch_file.exists():
                    entry["error"] = "no patch file"
                    detailed_results.append(entry)
                    continue

                patch_text = patch_file.read_text(errors="replace")
                category = classify_bug_from_patch(patch_text)
                entry["category"] = category
                in_scope = category in GH_DETECTABLE
                entry["in_scope"] = in_scope

                if in_scope:
                    bugs_in_scope += 1
                    proj["in_scope"] += 1

                # Track category stats
                if category not in by_category:
                    by_category[category] = {"total": 0, "in_scope": 0,
                                             "baseline_detected": 0, "liquid_detected": 0}
                by_category[category]["total"] += 1
                if in_scope:
                    by_category[category]["in_scope"] += 1

                extracted = extract_buggy_functions_from_patch(patch_text)
                entry["functions_extracted"] = len(extracted)

                b_bugs = []
                l_bugs = []

                for func_name, func_body in extracted:
                    source = make_analyzable_function(func_name, func_body)
                    if source is None:
                        continue

                    b_bugs.extend(run_baseline_on_code(source))
                    l_bugs.extend(run_liquid_on_code(source))

                entry["baseline_bugs"] = b_bugs
                entry["liquid_bugs"] = l_bugs
                entry["baseline_detected"] = len(b_bugs) > 0
                entry["liquid_detected"] = len(l_bugs) > 0

                if len(b_bugs) > 0:
                    baseline_detected += 1
                    proj["baseline_detected"] += 1
                    if category in by_category:
                        by_category[category]["baseline_detected"] += 1

                if len(l_bugs) > 0:
                    liquid_detected += 1
                    proj["liquid_detected"] += 1
                    if category in by_category:
                        by_category[category]["liquid_detected"] += 1

            except Exception as e:
                entry["error"] = str(e)

            detailed_results.append(entry)

        by_project[project] = proj
        print(f"  {project}: {proj['total']} bugs, {proj['in_scope']} in-scope, "
              f"baseline={proj['baseline_detected']}, liquid={proj['liquid_detected']}")

    baseline_recall = baseline_detected / total_bugs if total_bugs > 0 else 0
    liquid_recall = liquid_detected / total_bugs if total_bugs > 0 else 0
    improvement = liquid_detected / baseline_detected if baseline_detected > 0 else float('inf')

    output = {
        "summary": {
            "total_bugs_examined": total_bugs,
            "bugs_in_scope": bugs_in_scope,
            "baseline_detected": baseline_detected,
            "liquid_detected": liquid_detected,
            "baseline_recall": round(baseline_recall, 4),
            "liquid_recall": round(liquid_recall, 4),
            "improvement_factor": round(improvement, 2) if improvement != float('inf') else "inf",
        },
        "by_project": by_project,
        "by_category": by_category,
        "detailed_results": detailed_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "liquid_bugsinpy_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BugsInPy: Baseline vs Liquid Type Inference")
    print(f"{'='*60}")
    print(f"Total bugs examined:     {total_bugs}")
    print(f"Bugs in scope:           {bugs_in_scope}")
    print(f"Baseline detected:       {baseline_detected}")
    print(f"Liquid detected:         {liquid_detected}")
    print(f"Baseline recall:         {baseline_recall:.2%}")
    print(f"Liquid recall:           {liquid_recall:.2%}")
    print(f"Improvement factor:      {improvement:.2f}x" if improvement != float('inf') else "Improvement: inf (baseline=0)")
    print(f"\nBy category:")
    for cat, cs in sorted(by_category.items()):
        print(f"  {cat:25s}: baseline={cs['baseline_detected']:3d}, liquid={cs['liquid_detected']:3d} / {cs['total']:3d}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
