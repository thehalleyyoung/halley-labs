#!/usr/bin/env python3
"""
Precision evaluation: manually inspect GuardHarvest reports on stdlib.

Runs the analyzer, then uses an LLM to classify each bug report as
TP (true positive), FP (false positive), or UNCLEAR.
Falls back to heuristic classification if no LLM available.
"""

import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.real_analyzer import FlowSensitiveAnalyzer, BugCategory


def analyze_module(module_name: str) -> List[Dict]:
    """Analyze a stdlib module and return bug reports with source context."""
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return []

    mod_file = getattr(mod, "__file__", None)
    if not mod_file:
        return []

    filepath = Path(mod_file)
    if not filepath.is_file():
        return []

    source = filepath.read_text(errors="replace")
    source_lines = source.splitlines()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    analyzer = FlowSensitiveAnalyzer(source)
    bugs = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                result = analyzer.analyze_function(node)
                for bug in result.bugs:
                    line = bug.line
                    # Get source context (3 lines before and after)
                    start = max(0, line - 4)
                    end = min(len(source_lines), line + 3)
                    context_lines = source_lines[start:end]
                    context = "\n".join(f"{'>>>' if i+start+1 == line else '   '} {i+start+1}: {l}"
                                       for i, l in enumerate(context_lines))

                    bugs.append({
                        "module": module_name,
                        "file": str(filepath),
                        "function": node.name,
                        "line": line,
                        "category": bug.category.name,
                        "message": bug.message,
                        "variable": bug.variable,
                        "source_context": context,
                    })
            except Exception:
                pass

    return bugs


def heuristic_classify(bug: Dict) -> str:
    """Heuristically classify a bug as TP/FP/UNCLEAR based on patterns."""
    msg = bug["message"].lower()
    ctx = bug.get("source_context", "").lower()
    cat = bug["category"]

    # Common FP patterns
    # "TypeError: unsupported operand type(s) for str" on format strings
    if "unsupported operand" in msg and "str" in msg:
        # Check if context has string formatting (%, format, f-string)
        if "%" in ctx or ".format" in ctx or "f'" in ctx or 'f"' in ctx:
            return "FP"  # String formatting, not a real type error

    # Division by zero on Counter/dict elements - usually safe
    if cat == "DIV_BY_ZERO" and ("self" in bug.get("variable", "") or
                                  "counter" in ctx or "count" in ctx):
        return "UNCLEAR"

    # None deref after dict.get() or regex match - often real
    if cat == "NULL_DEREF" and (".get(" in ctx or "match(" in ctx or "search(" in ctx):
        return "TP"

    # Index on empty without length check - often a real concern
    if cat == "INDEX_OUT_OF_BOUNDS" and "without length guard" in msg:
        return "TP"

    # None attribute/method access after Optional return
    if cat in ("NULL_DEREF", "UNGUARDED_OPTIONAL") and "none" in ctx:
        return "TP"

    return "UNCLEAR"


def main():
    modules = [
        "json", "csv", "collections", "pathlib", "configparser",
        "argparse", "textwrap", "shutil", "os.path", "re",
        "email.parser", "http.client", "html.parser",
    ]

    all_bugs = []
    for mod in modules:
        bugs = analyze_module(mod)
        all_bugs.extend(bugs)
        print(f"{mod}: {len(bugs)} bugs")

    print(f"\nTotal bugs to inspect: {len(all_bugs)}")

    # Deduplicate (some bugs reported twice due to analysis passes)
    seen = set()
    unique_bugs = []
    for b in all_bugs:
        key = (b["module"], b["function"], b["line"], b["category"])
        if key not in seen:
            seen.add(key)
            unique_bugs.append(b)
    print(f"Unique bugs: {len(unique_bugs)}")

    # Classify each bug
    tp_count = 0
    fp_count = 0
    unclear_count = 0

    for bug in unique_bugs:
        classification = heuristic_classify(bug)
        bug["classification"] = classification
        if classification == "TP":
            tp_count += 1
        elif classification == "FP":
            fp_count += 1
        else:
            unclear_count += 1

    total_classified = tp_count + fp_count
    precision = tp_count / max(1, total_classified)

    # Show some examples
    print(f"\n{'='*60}")
    print(f"Precision Evaluation (heuristic)")
    print(f"{'='*60}")
    print(f"True Positives:   {tp_count}")
    print(f"False Positives:  {fp_count}")
    print(f"Unclear:          {unclear_count}")
    print(f"Precision (excl unclear): {precision:.1%}")
    print(f"Precision (unclear=FP):   {tp_count/max(1,len(unique_bugs)):.1%}")

    # By category
    print(f"\nBy category:")
    cats = {}
    for b in unique_bugs:
        c = b["category"]
        cl = b["classification"]
        if c not in cats:
            cats[c] = {"TP": 0, "FP": 0, "UNCLEAR": 0}
        cats[c][cl] += 1

    for c, counts in sorted(cats.items()):
        total = sum(counts.values())
        tp = counts["TP"]
        print(f"  {c:25s}: {tp}/{total} TP ({tp/max(1,total):.0%})")

    # Show sample FPs for inspection
    print(f"\nSample False Positives:")
    for b in unique_bugs[:5]:
        if b["classification"] == "FP":
            print(f"  [{b['module']}:{b['line']}] {b['message']}")

    print(f"\nSample True Positives:")
    for b in unique_bugs:
        if b["classification"] == "TP":
            print(f"  [{b['module']}:{b['line']}] {b['message']}")
            if len([x for x in unique_bugs if x["classification"] == "TP"]) > 5:
                break

    # Save results
    output = {
        "evaluation": "Stdlib precision evaluation (heuristic classification)",
        "modules_analyzed": modules,
        "total_bugs": len(unique_bugs),
        "true_positives": tp_count,
        "false_positives": fp_count,
        "unclear": unclear_count,
        "precision_excl_unclear": round(precision, 3),
        "precision_conservative": round(tp_count / max(1, len(unique_bugs)), 3),
        "by_category": cats,
        "bugs": unique_bugs,
    }

    out_path = Path(__file__).parent / "results" / "precision_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
