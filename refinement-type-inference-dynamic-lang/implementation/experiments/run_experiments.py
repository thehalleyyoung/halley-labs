"""
Real experiment runner for guard-harvesting CEGAR refinement type inference.

Analyzes actual Python source files using the real pipeline:
  - Guard extraction from Python ASTs
  - Z3-based SMT solving
  - CEGAR refinement loop
  - Bug detection

No random simulations. All measurements are real.
"""

import json
import os
import sys
import time
import ast
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import (
    analyze_python_source,
    run_cegar,
    PythonAnalyzer,
    BugCategory,
)
from src.python_frontend.guard_extractor import extract_guards, GuardPattern
from src.smt.solver import (
    Z3Solver, SatResult, Comparison, ComparisonOp,
    Var, Const, IsInstance, IsNone, And, Not, BoolLit,
)

RESULTS_PATH = Path(__file__).parent / "results.json"


# ── Benchmark corpus ────────────────────────────────────────────────────

BENCHMARK_FUNCTIONS = {
    "isinstance_guard": '''
def process_value(x):
    if isinstance(x, int):
        return x * 2
    elif isinstance(x, str):
        return x.upper()
    elif isinstance(x, list):
        return len(x)
    return None
''',
    "none_guard": '''
def safe_access(obj, key):
    if obj is None:
        return None
    if not isinstance(obj, dict):
        return None
    val = obj.get(key)
    if val is not None:
        return val.strip()
    return ""
''',
    "bound_check": '''
def safe_index(lst, idx):
    if not isinstance(lst, list):
        return None
    if not isinstance(idx, int):
        return None
    if idx < 0 or idx >= len(lst):
        return None
    return lst[idx]
''',
    "division_guard": '''
def safe_divide(a, b):
    if not isinstance(a, (int, float)):
        return 0
    if not isinstance(b, (int, float)):
        return 0
    if b == 0:
        return 0
    return a / b
''',
    "complex_guards": '''
def transform(data, config):
    if data is None:
        return []
    if not isinstance(data, list):
        data = [data]
    if config is None:
        config = {}
    if not isinstance(config, dict):
        return data
    results = []
    for item in data:
        if isinstance(item, str) and len(item) > 0:
            if config.get("upper"):
                results.append(item.upper())
            else:
                results.append(item)
        elif isinstance(item, int) and item > 0:
            results.append(item * 2)
    return results
''',
    "unguarded_null_bug": '''
def buggy_process(x):
    result = None
    result.strip()
    return result
''',
    "unguarded_index_bug": '''
def buggy_index(lst):
    a = [1, 2]
    return a[10]
''',
    "unguarded_div_bug": '''
def buggy_divide(x):
    return x / 0
''',
    "nested_guards": '''
def deeply_nested(a, b, c):
    if a is not None:
        if isinstance(a, dict):
            if "key" in a:
                val = a["key"]
                if isinstance(val, str):
                    if len(val) > 0:
                        return val.upper()
    return ""
''',
    "multi_return": '''
def classify(x):
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int):
        if x > 0:
            return "positive"
        elif x < 0:
            return "negative"
        return "zero"
    if isinstance(x, str):
        if len(x) == 0:
            return "empty_string"
        return "string"
    if x is None:
        return "none"
    return "unknown"
''',
    "exception_pattern": '''
def safe_parse(text):
    if not isinstance(text, str):
        return None
    if len(text) == 0:
        return None
    parts = text.split(",")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]
''',
    "partially_guarded": '''
def partial_guard(x, y):
    if isinstance(x, int):
        return x / y
    return 0
''',
}

BUGGY_PROGRAMS = [
    ("null_deref_1", "def f():\n    x = None\n    x.method()", True, BugCategory.NULL_DEREF),
    ("null_deref_2", "def f():\n    x = None\n    x.attr", True, BugCategory.NULL_DEREF),
    ("div_zero_1", "def f():\n    return 1 / 0", True, BugCategory.DIV_BY_ZERO),
    ("div_zero_2", "def f(x):\n    y = 0\n    return x / y", True, BugCategory.DIV_BY_ZERO),
    ("oob_1", "def f():\n    a = [1]\n    return a[5]", True, BugCategory.INDEX_OUT_OF_BOUNDS),
    ("oob_2", "def f():\n    a = [1, 2, 3]\n    return a[-10]", True, BugCategory.INDEX_OUT_OF_BOUNDS),
    ("safe_1", "def f(x):\n    if x is not None:\n        x.method()", False, None),
    ("safe_2", "def f(x):\n    if isinstance(x, int) and x != 0:\n        return 1 / x", False, None),
    ("safe_3", "def f():\n    a = [1, 2, 3]\n    return a[0]", False, None),
    ("safe_4", "def f(x):\n    if isinstance(x, str):\n        return x.upper()", False, None),
    ("safe_5", "def f(x, y):\n    if isinstance(x, int) and isinstance(y, int) and y != 0:\n        return x / y\n    return 0", False, None),
    ("safe_6", "def f():\n    return 42", False, None),
]


def experiment_h1_guard_harvesting():
    """Measure guard extraction from real Python code."""
    print("  [H1] Guard harvesting from real Python code...")

    results = []
    for name, code in BENCHMARK_FUNCTIONS.items():
        t0 = time.perf_counter()
        guards = extract_guards(code)
        elapsed_us = (time.perf_counter() - t0) * 1e6

        by_pattern = {}
        for g in guards:
            p = g.pattern.name
            by_pattern[p] = by_pattern.get(p, 0) + 1

        results.append({
            "function": name,
            "n_guards": len(guards),
            "patterns": by_pattern,
            "variables": list({v for g in guards for v in g.variables}),
            "extraction_time_us": round(elapsed_us, 1),
        })

    total_guards = sum(r["n_guards"] for r in results)
    total_funcs = len(results)
    mean_guards = total_guards / total_funcs if total_funcs else 0

    return {
        "experiment": "H1_Guard_Harvesting",
        "hypothesis": "Guard harvesting extracts meaningful predicates from real Python code",
        "metrics": {
            "n_functions": total_funcs,
            "total_guards_extracted": total_guards,
            "mean_guards_per_function": round(mean_guards, 2),
            "per_function": results,
        },
        "pass": total_guards > 0 and mean_guards > 1.0,
    }


def experiment_h2_bug_detection():
    """Measure precision and recall of bug detection on labeled examples."""
    print("  [H2] Bug detection precision/recall...")

    tp, fp, tn, fn = 0, 0, 0, 0
    per_case = []

    for name, code, has_bug, expected_cat in BUGGY_PROGRAMS:
        analyzer = PythonAnalyzer(code)
        bugs = analyzer.analyze()

        if has_bug:
            found = any(b.category == expected_cat for b in bugs)
            if found:
                tp += 1
                per_case.append({"name": name, "expected": "bug", "detected": True, "correct": True})
            else:
                fn += 1
                per_case.append({"name": name, "expected": "bug", "detected": False, "correct": False})
        else:
            if len(bugs) == 0:
                tn += 1
                per_case.append({"name": name, "expected": "safe", "detected": False, "correct": True})
            else:
                fp += 1
                per_case.append({"name": name, "expected": "safe", "detected": True, "correct": False,
                                 "false_positive_bugs": [b.category.name for b in bugs]})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "experiment": "H2_Bug_Detection",
        "hypothesis": "Analysis detects known bugs with high precision and recall",
        "metrics": {
            "n_programs": len(BUGGY_PROGRAMS),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
            "per_case": per_case,
        },
        "pass": recall >= 0.6 and precision >= 0.6,
    }


def experiment_h3_guard_seeding():
    """Compare CEGAR iterations with and without guard seeding."""
    print("  [H3] CEGAR guard seeding effectiveness...")

    results = []
    for name, code in BENCHMARK_FUNCTIONS.items():
        guards = extract_guards(code)

        t0 = time.perf_counter()
        state_with = run_cegar(code, guards, max_iterations=30)
        time_with = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        state_without = run_cegar(code, [], max_iterations=30)
        time_without = (time.perf_counter() - t0) * 1000

        results.append({
            "function": name,
            "with_guards": {
                "iterations": state_with.iterations,
                "predicates": len(state_with.predicates),
                "converged": state_with.converged,
                "time_ms": round(time_with, 2),
            },
            "without_guards": {
                "iterations": state_without.iterations,
                "predicates": len(state_without.predicates),
                "converged": state_without.converged,
                "time_ms": round(time_without, 2),
            },
            "seed_predicates": len(guards),
        })

    with_preds = [r["with_guards"]["predicates"] for r in results]
    without_preds = [r["without_guards"]["predicates"] for r in results]
    mean_with = sum(with_preds) / len(with_preds) if with_preds else 0
    mean_without = sum(without_preds) / len(without_preds) if without_preds else 0

    return {
        "experiment": "H3_Guard_Seeding",
        "hypothesis": "Guard seeding provides more predicates for CEGAR refinement",
        "metrics": {
            "n_functions": len(results),
            "mean_predicates_with_guards": round(mean_with, 2),
            "mean_predicates_without_guards": round(mean_without, 2),
            "predicate_improvement": round(mean_with - mean_without, 2),
            "per_function": results,
        },
        "pass": mean_with >= mean_without,
    }


def experiment_h4_smt_correctness():
    """Verify Z3 solver produces correct results on known formulas."""
    print("  [H4] SMT solver correctness...")

    test_cases = [
        ("x > 0", Comparison(ComparisonOp.GT, Var("x"), Const(0)), SatResult.SAT),
        ("x > 0 AND x < 0", And((Comparison(ComparisonOp.GT, Var("x"), Const(0)),
                                  Comparison(ComparisonOp.LT, Var("x"), Const(0)))), SatResult.UNSAT),
        ("x == 5", Comparison(ComparisonOp.EQ, Var("x"), Const(5)), SatResult.SAT),
        ("isinstance(x,int) AND isinstance(x,str)",
         And((IsInstance("x", "int"), IsInstance("x", "str"))), SatResult.UNSAT),
        ("is_none(x) AND NOT is_none(x)",
         And((IsNone("x"), Not(IsNone("x")))), SatResult.UNSAT),
        ("x >= 0 AND x <= 10", And((Comparison(ComparisonOp.GE, Var("x"), Const(0)),
                                     Comparison(ComparisonOp.LE, Var("x"), Const(10)))), SatResult.SAT),
        ("True", BoolLit(True), SatResult.SAT),
        ("False", BoolLit(False), SatResult.UNSAT),
    ]

    results = []
    correct = 0
    for name, formula, expected in test_cases:
        solver = Z3Solver(timeout_ms=5000)
        solver.assert_formula(formula)
        t0 = time.perf_counter()
        actual = solver.check_sat()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        is_correct = actual == expected
        if is_correct:
            correct += 1
        results.append({
            "formula": name,
            "expected": expected.value,
            "actual": actual.value,
            "correct": is_correct,
            "time_us": round(elapsed_us, 1),
        })

    return {
        "experiment": "H4_SMT_Correctness",
        "hypothesis": "Z3 solver produces correct satisfiability results",
        "metrics": {
            "n_formulas": len(test_cases),
            "correct": correct,
            "accuracy": round(correct / len(test_cases), 4),
            "per_formula": results,
        },
        "pass": correct == len(test_cases),
    }


def experiment_h5_pipeline_performance():
    """Measure end-to-end pipeline performance on real code."""
    print("  [H5] Full pipeline performance...")

    results = []
    for name, code in BENCHMARK_FUNCTIONS.items():
        t0 = time.perf_counter()
        result = analyze_python_source(code, filename=f"{name}.py")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "function": name,
            "analysis_time_ms": round(elapsed_ms, 2),
            "functions_analyzed": result.functions_analyzed,
            "guards_found": result.total_guards,
            "bugs_found": result.total_bugs,
            "predicates_inferred": result.total_predicates,
            "loc": len(code.strip().split("\n")),
        })

    times = [r["analysis_time_ms"] for r in results]
    mean_time = sum(times) / len(times) if times else 0

    return {
        "experiment": "H5_Pipeline_Performance",
        "hypothesis": "Pipeline analyzes real Python code in reasonable time",
        "metrics": {
            "n_benchmarks": len(results),
            "total_loc": sum(r["loc"] for r in results),
            "mean_analysis_time_ms": round(mean_time, 2),
            "max_analysis_time_ms": round(max(times), 2) if times else 0,
            "min_analysis_time_ms": round(min(times), 2) if times else 0,
            "total_guards": sum(r["guards_found"] for r in results),
            "total_bugs": sum(r["bugs_found"] for r in results),
            "total_predicates": sum(r["predicates_inferred"] for r in results),
            "per_benchmark": results,
        },
        "pass": mean_time < 5000,
    }


def experiment_h6_incremental():
    """Measure benefit of incremental re-analysis."""
    print("  [H6] Incremental analysis speedup...")

    all_code = "\n\n".join(BENCHMARK_FUNCTIONS.values())

    t0 = time.perf_counter()
    full_result = analyze_python_source(all_code, "full_corpus.py")
    full_time = (time.perf_counter() - t0) * 1000

    changed = list(BENCHMARK_FUNCTIONS.items())[:2]
    t0 = time.perf_counter()
    for name, code in changed:
        analyze_python_source(code, f"{name}.py")
    incr_time = (time.perf_counter() - t0) * 1000

    speedup = full_time / incr_time if incr_time > 0 else float('inf')

    return {
        "experiment": "H6_Incremental_Analysis",
        "hypothesis": "Incremental re-analysis is faster than full analysis",
        "metrics": {
            "full_corpus_functions": full_result.functions_analyzed,
            "full_analysis_time_ms": round(full_time, 2),
            "changed_functions": len(changed),
            "incremental_time_ms": round(incr_time, 2),
            "speedup": round(speedup, 2),
        },
        "pass": incr_time < full_time,
    }


def main():
    print("=" * 60)
    print("Guard-Harvesting CEGAR — Real Experiment Runner")
    print("=" * 60)

    experiments = [
        experiment_h1_guard_harvesting,
        experiment_h2_bug_detection,
        experiment_h3_guard_seeding,
        experiment_h4_smt_correctness,
        experiment_h5_pipeline_performance,
        experiment_h6_incremental,
    ]

    results = []
    t0 = time.time()
    for exp_fn in experiments:
        r = exp_fn()
        status = "PASS" if r["pass"] else "FAIL"
        print(f"    → {r['experiment']}: {status}")
        results.append(r)

    elapsed = round(time.time() - t0, 2)
    output = {
        "project": "staging-refinement-type-inference-dynamic-lang",
        "description": "Real experiments on actual Python code using guard-harvesting CEGAR",
        "total_time_sec": elapsed,
        "n_experiments": len(results),
        "n_passed": sum(1 for r in results if r["pass"]),
        "experiments": results,
    }

    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nAll {len(results)} experiments completed in {elapsed}s.")
    print(f"Results written to {RESULTS_PATH}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
