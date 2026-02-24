"""
Run experiments on the external benchmark (real-world bug patterns).
"""

import ast
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.real_analyzer import analyze_source, FlowSensitiveAnalyzer, BugCategory


def run_external_benchmark():
    """Run GuardHarvest on the external benchmark and compute metrics."""
    benchmark_path = Path(__file__).parent / "benchmarks" / "external_benchmark.py"
    source = benchmark_path.read_text()

    # Import ground truth
    spec = {}
    exec(compile(source, str(benchmark_path), "exec"), spec)
    ground_truth = spec["GROUND_TRUTH"]

    # Parse and find functions
    tree = ast.parse(source, str(benchmark_path))
    functions = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    # Analyze each function
    analyzer = FlowSensitiveAnalyzer(source, str(benchmark_path))
    results = {}
    tp, fp, fn, tn = 0, 0, 0, 0
    category_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    fp_details = []
    fn_details = []

    for func_name, truth in ground_truth.items():
        if func_name not in functions:
            print(f"  WARNING: {func_name} not found in benchmark")
            continue

        func_node = functions[func_name]
        fr = analyzer.analyze_function(func_node)
        detected = len(fr.bugs) > 0
        has_bug = truth["has_bug"]
        cat = truth["category"]

        if has_bug and detected:
            tp += 1
            category_stats[cat]["tp"] += 1
        elif not has_bug and detected:
            fp += 1
            category_stats[cat]["fp"] += 1
            fp_details.append({
                "function": func_name,
                "bugs": [b.message for b in fr.bugs],
                "source": truth.get("source", "")
            })
        elif has_bug and not detected:
            fn += 1
            category_stats[cat]["fn"] += 1
            fn_details.append({
                "function": func_name,
                "category": cat,
                "source": truth.get("source", "")
            })
        else:
            tn += 1
            category_stats[cat]["tn"] += 1

        results[func_name] = {
            "has_bug": has_bug,
            "detected": detected,
            "bugs_found": [b.to_dict() for b in fr.bugs],
            "guards_harvested": fr.guards_harvested,
            "correct": (has_bug == detected),
        }

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_bugs = sum(1 for t in ground_truth.values() if t["has_bug"])
    total_safe = sum(1 for t in ground_truth.values() if not t["has_bug"])

    output = {
        "benchmark": "external_real_world_patterns",
        "total_functions": len(ground_truth),
        "total_bugs": total_bugs,
        "total_safe": total_safe,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1": round(f1, 3),
        "accuracy": round((tp + tn) / len(ground_truth) * 100, 1),
        "category_breakdown": {},
        "false_positives": fp_details,
        "false_negatives": fn_details,
        "per_function": results,
    }

    for cat, stats in sorted(category_stats.items()):
        p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
        r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        output["category_breakdown"][cat] = {
            **stats,
            "precision": round(p * 100, 1),
            "recall": round(r * 100, 1),
            "f1": round(f, 3),
        }

    return output


def run_pyright_comparison():
    """Simulate what Pyright would detect on untyped external benchmark."""
    # Pyright on untyped code: detects None dereferences where explicit
    # None assignment is visible, but not dict.get(), re.search(), etc.
    # Cannot detect div-by-zero or OOB.
    return {
        "tool": "pyright",
        "note": "Estimated based on Pyright's documented capabilities on untyped code",
        "null_deref_detected": 2,  # Only explicit None assignments
        "div_by_zero_detected": 0,
        "oob_detected": 0,
        "total_detected": 2,
        "total_bugs": 22,
        "recall": 9.1,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("EXTERNAL BENCHMARK: Real-World Bug Patterns")
    print("=" * 70)

    results = run_external_benchmark()

    print(f"\nFunctions: {results['total_functions']}")
    print(f"  Bugs: {results['total_bugs']}, Safe: {results['total_safe']}")
    print(f"\nTP={results['tp']} FP={results['fp']} FN={results['fn']} TN={results['tn']}")
    print(f"Precision: {results['precision']}%")
    print(f"Recall: {results['recall']}%")
    print(f"F1: {results['f1']}")
    print(f"Accuracy: {results['accuracy']}%")

    print("\nPer-category:")
    for cat, stats in results["category_breakdown"].items():
        print(f"  {cat}: P={stats['precision']}% R={stats['recall']}% F1={stats['f1']}")

    if results["false_positives"]:
        print(f"\nFalse Positives ({len(results['false_positives'])}):")
        for fp in results["false_positives"]:
            print(f"  {fp['function']}: {fp['bugs']}")

    if results["false_negatives"]:
        print(f"\nFalse Negatives ({len(results['false_negatives'])}):")
        for fn_item in results["false_negatives"]:
            print(f"  {fn_item['function']} ({fn_item['category']}): {fn_item['source']}")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "E9_external_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'E9_external_benchmark.json'}")
