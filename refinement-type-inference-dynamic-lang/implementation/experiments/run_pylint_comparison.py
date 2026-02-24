"""
Experiment: Compare GuardHarvest with Pylint on the benchmark suite.
Provides a fair comparison against a syntactic pattern-matching tool.
"""

import sys
import ast
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.benchmarks.benchmark_suite import GROUND_TRUTH
from src.real_analyzer import analyze_source


def run_pylint_comparison():
    """Run Pylint on the benchmark and compare with GuardHarvest."""
    bench_path = Path(__file__).parent / "benchmarks" / "benchmark_suite.py"

    # Get function line ranges from AST
    source = bench_path.read_text()
    tree = ast.parse(source)
    func_ranges = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in GROUND_TRUTH:
            func_ranges[node.name] = (node.lineno, node.end_lineno or node.lineno + 20)

    # Run Pylint (errors only)
    result = subprocess.run(
        ['pylint', str(bench_path), '--output-format=json'],
        capture_output=True, text=True
    )
    pylint_issues = json.loads(result.stdout) if result.stdout.strip() else []

    pylint_bugs_by_func = {}
    for issue in pylint_issues:
        line = issue.get('line', 0)
        msg_id = issue.get('message-id', '')
        if msg_id[0] != 'E':
            continue
        for fname, (start, end) in func_ranges.items():
            if start <= line <= end:
                pylint_bugs_by_func.setdefault(fname, []).append(msg_id)
                break

    # Calculate Pylint metrics
    tp = fp = fn = tn = 0
    for fname, info in GROUND_TRUTH.items():
        has_bug = info['has_bug']
        detected = fname in pylint_bugs_by_func
        if has_bug and detected: tp += 1
        elif not has_bug and detected: fp += 1
        elif has_bug and not detected: fn += 1
        else: tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    pylint_result = {
        "tool": "Pylint",
        "version": "4.0.4",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "detected_functions": list(pylint_bugs_by_func.keys()),
        "note": "Errors only (E-codes). Pylint is a syntactic pattern matcher; "
                "it detects type errors (no-member) but not null deref, div-by-zero, or OOB."
    }

    print(f"Pylint: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision: {prec:.1%} Recall: {rec:.1%} F1: {f1:.3f}")
    print(f"  Detected: {list(pylint_bugs_by_func.keys())}")

    # Save result
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "E8_pylint_comparison.json", "w") as f:
        json.dump(pylint_result, f, indent=2)

    return pylint_result


if __name__ == "__main__":
    run_pylint_comparison()
