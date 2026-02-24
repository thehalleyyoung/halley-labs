"""
Syntactic baseline for bug detection (addresses critique W9/A7).

A simple pattern-matching baseline that flags:
1. Division where divisor is a function parameter without a preceding != 0 guard
2. .get() call result used without a None check
3. Explicit None assignment followed by attribute access

This does NOT use flow-sensitive analysis, abstract interpretation, or interval
arithmetic. It demonstrates the added value of GuardHarvest's abstract domain.
"""

import ast
import json
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

class SyntacticBaseline(ast.NodeVisitor):
    """Simple pattern-matching bug detector."""

    def __init__(self):
        self.bugs: List[Dict[str, Any]] = []
        self._func_name = ""
        self._params: set = set()
        self._none_vars: set = set()
        self._get_vars: set = set()
        self._guarded_vars: set = set()
        self._nonzero_guarded: set = set()

    def analyze_function(self, func: ast.FunctionDef) -> List[Dict[str, Any]]:
        self.bugs = []
        self._func_name = func.name
        self._params = {a.arg for a in func.args.args}
        self._none_vars = set()
        self._get_vars = set()
        self._guarded_vars = set()
        self._nonzero_guarded = set()

        # Two-pass: first find guards, then find bugs
        for stmt in ast.walk(func):
            self._collect_guards(stmt)
        for stmt in ast.walk(func):
            self._find_bugs(stmt)
        return self.bugs

    def _collect_guards(self, node):
        """Collect guard patterns (is not None, != 0, isinstance, truthiness in if)."""
        if isinstance(node, ast.If):
            test = node.test
            # if x is not None / if x is None: return
            if isinstance(test, ast.Compare) and len(test.ops) == 1:
                if isinstance(test.ops[0], ast.IsNot):
                    if isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None:
                        if isinstance(test.left, ast.Name):
                            self._guarded_vars.add(test.left.id)
                elif isinstance(test.ops[0], ast.Is):
                    if isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None:
                        if isinstance(test.left, ast.Name):
                            # if x is None: return -> guard in else
                            if node.body and isinstance(node.body[-1], (ast.Return, ast.Raise)):
                                self._guarded_vars.add(test.left.id)
                elif isinstance(test.ops[0], ast.NotEq):
                    if isinstance(test.left, ast.Name) and isinstance(test.comparators[0], ast.Constant):
                        if test.comparators[0].value == 0:
                            self._nonzero_guarded.add(test.left.id)
                elif isinstance(test.ops[0], (ast.Gt, ast.GtE)):
                    if isinstance(test.left, ast.Name) and isinstance(test.comparators[0], ast.Constant):
                        if isinstance(test.comparators[0].value, (int, float)) and test.comparators[0].value >= 0:
                            self._nonzero_guarded.add(test.left.id)
            # if x: (truthiness)
            elif isinstance(test, ast.Name):
                self._guarded_vars.add(test.id)
                self._nonzero_guarded.add(test.id)
            # isinstance check
            elif isinstance(test, ast.Call) and isinstance(test.func, ast.Name):
                if test.func.id == "isinstance" and test.args and isinstance(test.args[0], ast.Name):
                    self._guarded_vars.add(test.args[0].id)
        # Assert guards
        elif isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1:
                if isinstance(test.ops[0], ast.IsNot):
                    if isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value is None:
                        if isinstance(test.left, ast.Name):
                            self._guarded_vars.add(test.left.id)

    def _find_bugs(self, node):
        """Find bug patterns."""
        # Pattern 1: Division by parameter without guard
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            divisor = node.right
            if isinstance(divisor, ast.Name):
                if divisor.id in self._params and divisor.id not in self._nonzero_guarded:
                    self.bugs.append({
                        "category": "DIV_BY_ZERO",
                        "line": node.lineno,
                        "message": f"Division by parameter '{divisor.id}' without != 0 guard",
                        "function": self._func_name,
                        "variable": divisor.id,
                    })
                elif divisor.id not in self._params and divisor.id not in self._nonzero_guarded:
                    # Division by local var without guard - only flag if it's a known-zero pattern
                    pass
            elif isinstance(divisor, ast.Constant) and divisor.value == 0:
                self.bugs.append({
                    "category": "DIV_BY_ZERO",
                    "line": node.lineno,
                    "message": "Division by literal zero",
                    "function": self._func_name,
                    "variable": "0",
                })
        # Pattern 1b: Augmented assignment division
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if isinstance(node.value, ast.Name):
                if node.value.id in self._params and node.value.id not in self._nonzero_guarded:
                    self.bugs.append({
                        "category": "DIV_BY_ZERO",
                        "line": node.lineno,
                        "message": f"Division by parameter '{node.value.id}' without guard",
                        "function": self._func_name,
                        "variable": node.value.id,
                    })

        # Pattern 2: x = None; ... x.attr (without guard)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and node.value.value is None:
                    self._none_vars.add(target.id)
                elif isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Attribute) and node.value.func.attr in ("get", "pop"):
                        self._get_vars.add(target.id)
                    elif isinstance(node.value.func, ast.Name) and node.value.func.id not in (
                        "int", "float", "str", "bool", "list", "tuple", "dict", "set",
                        "len", "range", "sorted", "abs", "max", "min", "sum"):
                        # Unknown function call - could return None
                        pass

        # Pattern 3: Attribute access on None variable
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            var = node.value.id
            if var in self._none_vars and var not in self._guarded_vars:
                self.bugs.append({
                    "category": "NULL_DEREF",
                    "line": node.lineno,
                    "message": f"Attribute access on None variable '{var}'",
                    "function": self._func_name,
                    "variable": var,
                })
            elif var in self._get_vars and var not in self._guarded_vars:
                self.bugs.append({
                    "category": "UNGUARDED_OPTIONAL",
                    "line": node.lineno,
                    "message": f"Attribute access on .get() result '{var}' without None check",
                    "function": self._func_name,
                    "variable": var,
                })

        # Pattern 3b: Method call on None variable
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            var = node.func.value.id
            if var in self._none_vars and var not in self._guarded_vars:
                self.bugs.append({
                    "category": "NULL_DEREF",
                    "line": node.lineno,
                    "message": f"Method call on None variable '{var}'",
                    "function": self._func_name,
                    "variable": var,
                })
            elif var in self._get_vars and var not in self._guarded_vars:
                self.bugs.append({
                    "category": "UNGUARDED_OPTIONAL",
                    "line": node.lineno,
                    "message": f"Method call on .get() result '{var}' without None check",
                    "function": self._func_name,
                    "variable": var,
                })


def run_baseline_on_file(filepath: str, ground_truth: dict) -> dict:
    """Run syntactic baseline on a benchmark file with ground truth."""
    with open(filepath) as f:
        source = f.read()
    tree = ast.parse(source)

    baseline = SyntacticBaseline()
    results = {"per_function": [], "summary": {}}
    tp = fp = fn = tn = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ground_truth:
            bugs = baseline.analyze_function(node)
            gt = ground_truth[node.name]
            detected = len(bugs) > 0
            is_buggy = gt["buggy"]

            if detected and is_buggy:
                verdict = "TP"
                tp += 1
            elif detected and not is_buggy:
                verdict = "FP"
                fp += 1
            elif not detected and is_buggy:
                verdict = "FN"
                fn += 1
            else:
                verdict = "TN"
                tn += 1

            results["per_function"].append({
                "function": node.name,
                "buggy": is_buggy,
                "detected": detected,
                "verdict": verdict,
                "bugs_found": [b["category"] for b in bugs],
                "category": gt.get("category", "UNKNOWN"),
            })

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    results["summary"] = {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }
    return results


def build_ground_truth_from_benchmark(filepath: str) -> dict:
    """Extract ground truth from benchmark file's GROUND_TRUTH dict if available,
    or from E1_benchmark.json."""
    gt = {}
    results_path = os.path.join(os.path.dirname(filepath), "..", "results", "E1_benchmark.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        for entry in data["per_function"]:
            gt[entry["function"]] = {
                "buggy": entry["ground_truth_buggy"],
                "category": entry.get("category", "UNKNOWN"),
            }
    return gt


def build_external_ground_truth(filepath: str) -> dict:
    """Extract ground truth from external benchmark."""
    gt = {}
    results_path = os.path.join(os.path.dirname(filepath), "..", "results", "E9_external_benchmark.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        pf = data.get("per_function", {})
        if isinstance(pf, dict):
            for func_name, entry in pf.items():
                gt[func_name] = {
                    "buggy": entry.get("has_bug", False),
                    "category": entry.get("category", "UNKNOWN"),
                }
        else:
            for entry in pf:
                gt[entry["function"]] = {
                    "buggy": entry.get("ground_truth_buggy", entry.get("has_bug", False)),
                    "category": entry.get("category", "UNKNOWN"),
                }
    return gt


if __name__ == "__main__":
    base_dir = Path(__file__).parent

    # Run on author benchmark
    bench_file = base_dir / "benchmarks" / "benchmark_suite.py"
    gt = build_ground_truth_from_benchmark(str(bench_file))
    if gt:
        print("=" * 60)
        print("SYNTACTIC BASELINE: Author Benchmark (97 functions)")
        print("=" * 60)
        results = run_baseline_on_file(str(bench_file), gt)
        s = results["summary"]
        print(f"  TP={s['TP']} FP={s['FP']} FN={s['FN']} TN={s['TN']}")
        print(f"  Precision: {s['precision']*100:.1f}%")
        print(f"  Recall: {s['recall']*100:.1f}%")
        print(f"  F1: {s['f1_score']:.3f}")

        # Per-category breakdown
        cats = {}
        for r in results["per_function"]:
            c = r["category"]
            if c not in cats:
                cats[c] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
            cats[c][r["verdict"]] += 1
        print("\n  Per-category:")
        for c, v in sorted(cats.items()):
            total_p = v["TP"] + v["FP"]
            prec = v["TP"] / total_p if total_p > 0 else 0
            total_r = v["TP"] + v["FN"]
            rec = v["TP"] / total_r if total_r > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"    {c}: TP={v['TP']} FP={v['FP']} FN={v['FN']} Prec={prec*100:.0f}% Rec={rec*100:.0f}% F1={f1:.3f}")

        # Detailed FP/FN analysis
        print("\n  False Positives:")
        for r in results["per_function"]:
            if r["verdict"] == "FP":
                print(f"    {r['function']}: {r['bugs_found']}")
        print("\n  True Positives detected:")
        for r in results["per_function"]:
            if r["verdict"] == "TP":
                print(f"    {r['function']}: {r['bugs_found']}")

        # Save results
        out_path = base_dir / "results" / "E10_syntactic_baseline.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out_path}")

    # Run on external benchmark
    ext_file = base_dir / "benchmarks" / "external_benchmark.py"
    if ext_file.exists():
        ext_gt = build_external_ground_truth(str(ext_file))
        if ext_gt:
            print("\n" + "=" * 60)
            print("SYNTACTIC BASELINE: External Benchmark (39 functions)")
            print("=" * 60)
            ext_results = run_baseline_on_file(str(ext_file), ext_gt)
            s = ext_results["summary"]
            print(f"  TP={s['TP']} FP={s['FP']} FN={s['FN']} TN={s['TN']}")
            print(f"  Precision: {s['precision']*100:.1f}%")
            print(f"  Recall: {s['recall']*100:.1f}%")
            print(f"  F1: {s['f1_score']:.3f}")

            print("\n  True Positives detected:")
            for r in ext_results["per_function"]:
                if r["verdict"] == "TP":
                    print(f"    {r['function']}: {r['bugs_found']}")
            print("\n  False Positives:")
            for r in ext_results["per_function"]:
                if r["verdict"] == "FP":
                    print(f"    {r['function']}: {r['bugs_found']}")

            out_path = base_dir / "results" / "E11_syntactic_baseline_external.json"
            with open(out_path, "w") as f:
                json.dump(ext_results, f, indent=2)
            print(f"\n  Results saved to {out_path}")
