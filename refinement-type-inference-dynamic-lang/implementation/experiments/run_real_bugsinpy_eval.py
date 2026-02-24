#!/usr/bin/env python3
"""
Real-world BugsInPy evaluation for GuardHarvest.

Extracts buggy code from actual BugsInPy patches and runs GuardHarvest
on the pre-fix (buggy) code to evaluate detection capability on real bugs.
Only counts bugs in categories GuardHarvest targets:
  null_dereference, type_error, division_by_zero, index_out_of_bounds, attribute_error
"""

import ast
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.real_analyzer import FlowSensitiveAnalyzer


@dataclass
class BugInfo:
    project: str
    bug_id: str
    buggy_commit: str
    fixed_commit: str
    patch_file: str
    changed_files: List[str] = field(default_factory=list)
    bug_category: str = "unknown"  # classified from patch content


@dataclass
class EvalResult:
    project: str
    bug_id: str
    bug_category: str
    buggy_functions_extracted: int = 0
    guardharvest_bugs_found: int = 0
    guardharvest_bug_details: List[Dict] = field(default_factory=list)
    detected: bool = False
    detection_relevant: bool = False  # was this a bug type GH could detect?
    error: str = ""


BUGSINPY_ROOT = Path(__file__).parent / "BugsInPy" / "projects"

# Bug categories GuardHarvest targets
GH_DETECTABLE = {"null_dereference", "type_error", "division_by_zero",
                  "index_out_of_bounds", "attribute_error"}


def classify_bug_from_patch(patch_text: str) -> str:
    """Heuristically classify a bug from its patch content."""
    patch_lower = patch_text.lower()

    # Check for None-related fixes
    if any(p in patch_lower for p in [
        "is not none", "is none", "none", "nonetype",
        "attributeerror", "has no attribute"
    ]):
        return "null_dereference"

    # Type error patterns
    if any(p in patch_lower for p in [
        "isinstance", "typeerror", "type(", "type error"
    ]):
        return "type_error"

    # Division by zero
    if any(p in patch_lower for p in [
        "/ 0", "divisionerror", "zerodivision", "division by zero",
        "!= 0", "> 0"
    ]):
        return "division_by_zero"

    # Index errors
    if any(p in patch_lower for p in [
        "indexerror", "index out of", "out of range", "keyerror",
        "len(", "bounds"
    ]):
        return "index_out_of_bounds"

    # Attribute errors
    if any(p in patch_lower for p in [
        "getattr", "hasattr", "attribute"
    ]):
        return "attribute_error"

    return "logic_or_other"


def extract_buggy_functions_from_patch(patch_text: str) -> List[Tuple[str, str]]:
    """Extract (function_name, buggy_code) pairs from a unified diff.

    Reconstructs the pre-fix version of changed functions using diff context.
    """
    functions = []
    current_file = None
    current_func = None
    func_lines = []
    in_func = False
    base_indent = 0

    for line in patch_text.split("\n"):
        # Track file
        if line.startswith("--- a/"):
            current_file = line[6:]
            continue
        if line.startswith("+++ b/"):
            continue

        # Track function context from @@ headers
        if line.startswith("@@"):
            # e.g., @@ -599,7 +599,7 @@ def is_string_dtype(arr_or_dtype) -> bool:
            m = re.search(r'def\s+(\w+)\s*\(', line)
            if m:
                if in_func and func_lines:
                    code = "\n".join(func_lines)
                    if code.strip():
                        functions.append((current_func, code))
                current_func = m.group(1)
                func_lines = []
                in_func = True
                base_indent = 0
            continue

        if not in_func:
            continue

        # Skip added lines (they're in the fix, not the bug)
        if line.startswith("+"):
            continue

        # Include context lines and removed lines (which are the buggy code)
        if line.startswith("-"):
            actual_line = line[1:]  # remove the - prefix
        elif line.startswith(" "):
            actual_line = line[1:]  # remove the space prefix
        else:
            actual_line = line

        func_lines.append(actual_line)

    # Save last function
    if in_func and func_lines and current_func:
        code = "\n".join(func_lines)
        if code.strip():
            functions.append((current_func, code))

    return functions


def make_analyzable_function(func_name: str, func_body: str) -> Optional[str]:
    """Try to make a function body parseable by wrapping it properly."""
    # If it already starts with def, try as-is
    if func_body.strip().startswith("def "):
        try:
            ast.parse(func_body)
            return func_body
        except SyntaxError:
            pass

    # Wrap in a function definition
    indented = "\n".join("    " + l for l in func_body.split("\n"))
    wrapped = f"def {func_name}(*args, **kwargs):\n{indented}"
    try:
        ast.parse(wrapped)
        return wrapped
    except SyntaxError:
        pass

    # Try with minimal body
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


def run_guardharvest_on_code(source: str) -> List[Dict]:
    """Run GuardHarvest analyzer on source code, return list of bug dicts."""
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
                    all_bugs.append({
                        "category": bug.category if isinstance(bug.category, str) else bug.category.value if hasattr(bug.category, 'value') else str(bug.category),
                        "message": str(bug.message) if hasattr(bug, 'message') else str(bug),
                        "line": bug.line if hasattr(bug, 'line') else 0,
                    })
            except Exception as e:
                pass

    return all_bugs


def evaluate_bugsinpy():
    """Main evaluation: iterate over all BugsInPy bugs, extract code, run GH."""
    results = []
    stats = {
        "total_bugs": 0,
        "bugs_with_patches": 0,
        "functions_extracted": 0,
        "functions_parseable": 0,
        "bugs_in_scope": 0,  # bugs GH could potentially detect
        "bugs_detected": 0,
        "by_project": {},
        "by_category": {},
    }

    if not BUGSINPY_ROOT.exists():
        print(f"ERROR: BugsInPy root not found at {BUGSINPY_ROOT}")
        return

    projects = sorted(p.name for p in BUGSINPY_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(projects)} projects in BugsInPy")

    for project in projects:
        proj_dir = BUGSINPY_ROOT / project
        bugs_dir = proj_dir / "bugs"
        if not bugs_dir.exists():
            continue

        bug_dirs = sorted(
            (d for d in bugs_dir.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name)
        )

        proj_stats = {"total": 0, "in_scope": 0, "detected": 0}

        for bug_dir in bug_dirs:
            stats["total_bugs"] += 1
            proj_stats["total"] += 1

            bug_id = bug_dir.name
            patch_file = bug_dir / "bug_patch.txt"

            if not patch_file.exists():
                results.append(asdict(EvalResult(
                    project=project, bug_id=bug_id,
                    bug_category="unknown", error="no patch file"
                )))
                continue

            stats["bugs_with_patches"] += 1
            patch_text = patch_file.read_text(errors="replace")
            bug_category = classify_bug_from_patch(patch_text)

            # Extract functions from patch
            extracted = extract_buggy_functions_from_patch(patch_text)
            stats["functions_extracted"] += len(extracted)

            # Try to analyze each extracted function
            gh_bugs = []
            parseable_count = 0

            for func_name, func_body in extracted:
                source = make_analyzable_function(func_name, func_body)
                if source is None:
                    continue
                parseable_count += 1
                stats["functions_parseable"] += 1

                bugs_found = run_guardharvest_on_code(source)
                gh_bugs.extend(bugs_found)

            in_scope = bug_category in GH_DETECTABLE
            detected = len(gh_bugs) > 0

            if in_scope:
                stats["bugs_in_scope"] += 1
                proj_stats["in_scope"] += 1
                if detected:
                    stats["bugs_detected"] += 1
                    proj_stats["detected"] += 1

            # Track by category
            if bug_category not in stats["by_category"]:
                stats["by_category"][bug_category] = {"total": 0, "detected": 0}
            stats["by_category"][bug_category]["total"] += 1
            if detected:
                stats["by_category"][bug_category]["detected"] += 1

            result = EvalResult(
                project=project,
                bug_id=bug_id,
                bug_category=bug_category,
                buggy_functions_extracted=len(extracted),
                guardharvest_bugs_found=len(gh_bugs),
                guardharvest_bug_details=gh_bugs,
                detected=detected,
                detection_relevant=in_scope,
            )
            results.append(asdict(result))

        stats["by_project"][project] = proj_stats
        detected_str = f"{proj_stats['detected']}/{proj_stats['in_scope']}" if proj_stats['in_scope'] > 0 else "0/0"
        print(f"  {project}: {proj_stats['total']} bugs, {proj_stats['in_scope']} in-scope, {detected_str} detected")

    # Compute summary metrics
    if stats["bugs_in_scope"] > 0:
        recall_in_scope = stats["bugs_detected"] / stats["bugs_in_scope"]
    else:
        recall_in_scope = 0.0

    summary = {
        "evaluation": "BugsInPy Real-World Benchmark",
        "methodology": "Extract buggy functions from BugsInPy patches, run GuardHarvest on pre-fix code",
        "total_bugs_examined": stats["total_bugs"],
        "bugs_with_patches": stats["bugs_with_patches"],
        "functions_extracted": stats["functions_extracted"],
        "functions_parseable": stats["functions_parseable"],
        "bugs_in_guardharvest_scope": stats["bugs_in_scope"],
        "bugs_detected_by_guardharvest": stats["bugs_detected"],
        "recall_on_in_scope_bugs": round(recall_in_scope, 3),
        "by_project": stats["by_project"],
        "by_category": stats["by_category"],
        "note": "In-scope = null_dereference, type_error, division_by_zero, index_out_of_bounds, attribute_error",
        "honest_assessment": "These are REAL bugs from BugsInPy, not author-written reproductions.",
    }

    output = {
        "summary": summary,
        "results": results,
    }

    out_path = Path(__file__).parent / "results" / "bugsinpy_real_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BugsInPy Real-World Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total bugs examined:        {stats['total_bugs']}")
    print(f"Bugs with patches:          {stats['bugs_with_patches']}")
    print(f"Functions extracted:         {stats['functions_extracted']}")
    print(f"Functions parseable:         {stats['functions_parseable']}")
    print(f"Bugs in GH scope:           {stats['bugs_in_scope']}")
    print(f"Bugs detected by GH:        {stats['bugs_detected']}")
    print(f"Recall (in-scope):          {recall_in_scope:.1%}")
    print(f"\nBy category:")
    for cat, cs in sorted(stats["by_category"].items()):
        det_rate = cs['detected'] / cs['total'] if cs['total'] > 0 else 0
        print(f"  {cat:25s}: {cs['detected']:3d}/{cs['total']:3d} ({det_rate:.1%})")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    evaluate_bugsinpy()
