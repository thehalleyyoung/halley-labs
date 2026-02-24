"""
Run the real-world benchmark: functions extracted from actual OSS projects.

This evaluates GuardHarvest on code patterns from real bug-fix commits,
providing externally-valid precision/recall numbers.
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.real_analyzer import analyze_source, BugCategory
from experiments.benchmarks.realworld_benchmark import (
    REALWORLD_FUNCTIONS, GROUND_TRUTH, get_benchmark_stats,
)

BUG_CATEGORY_MAP = {
    "NULL_DEREF": {BugCategory.NULL_DEREF, BugCategory.UNGUARDED_OPTIONAL},
    "DIV_BY_ZERO": {BugCategory.DIV_BY_ZERO},
    "INDEX_OUT_OF_BOUNDS": {BugCategory.INDEX_OUT_OF_BOUNDS},
    "TYPE_ERROR": {BugCategory.TYPE_ERROR, BugCategory.ATTRIBUTE_ERROR},
}


def run_realworld_benchmark():
    stats = get_benchmark_stats()
    print("=" * 70)
    print("REAL-WORLD BENCHMARK: Functions from OSS Bug-Fix Commits")
    print("=" * 70)
    print(f"Total functions: {stats['total']} ({stats['buggy']} buggy, {stats['safe']} safe)")
    print(f"Categories: {stats['categories']}")
    print(f"Projects: {stats['projects']}")
    print()

    tp, fp, fn, tn = 0, 0, 0, 0
    cat_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    proj_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    details = []
    total_time = 0

    for name, source in REALWORLD_FUNCTIONS.items():
        gt = GROUND_TRUTH[name]
        expected_buggy = gt["buggy"]
        category = gt["category"]
        project = gt["project"]

        t0 = time.perf_counter()
        result = analyze_source(source, filename=f"{name}.py")
        elapsed = (time.perf_counter() - t0) * 1000
        total_time += elapsed

        # Collect bugs matching the expected category
        relevant_cats = BUG_CATEGORY_MAP.get(category, set())
        found_bugs = []
        for fr in result.function_results:
            for bug in fr.bugs:
                if bug.category in relevant_cats:
                    found_bugs.append(bug)

        detected = len(found_bugs) > 0

        if expected_buggy and detected:
            tp += 1
            cat_results[category]["tp"] += 1
            proj_results[project]["tp"] += 1
            verdict = "TP"
        elif expected_buggy and not detected:
            fn += 1
            cat_results[category]["fn"] += 1
            proj_results[project]["fn"] += 1
            verdict = "FN"
        elif not expected_buggy and detected:
            fp += 1
            cat_results[category]["fp"] += 1
            proj_results[project]["fp"] += 1
            verdict = "FP"
        else:
            tn += 1
            cat_results[category]["tn"] += 1
            proj_results[project]["tn"] += 1
            verdict = "TN"

        details.append({
            "name": name,
            "category": category,
            "project": project,
            "expected_buggy": expected_buggy,
            "detected": detected,
            "verdict": verdict,
            "bugs_found": [b.message for b in found_bugs],
            "time_ms": elapsed,
        })

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nOverall Results:")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Time:      {total_time:.1f}ms")

    print(f"\nPer-Category Results:")
    for cat in sorted(cat_results.keys()):
        r = cat_results[cat]
        p = r["tp"] / (r["tp"] + r["fp"]) if (r["tp"] + r["fp"]) > 0 else 0
        rc = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) > 0 else 0
        f = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0
        print(f"  {cat}: P={p:.1%} R={rc:.1%} F1={f:.3f} (TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']})")

    print(f"\nPer-Project Results:")
    for proj in sorted(proj_results.keys()):
        r = proj_results[proj]
        p = r["tp"] / (r["tp"] + r["fp"]) if (r["tp"] + r["fp"]) > 0 else 0
        rc = r["tp"] / (r["tp"] + r["fn"]) if (r["tp"] + r["fn"]) > 0 else 0
        f = 2 * p * rc / (p + rc) if (p + rc) > 0 else 0
        print(f"  {proj}: P={p:.1%} R={rc:.1%} F1={f:.3f}")

    # False negatives
    fn_list = [d for d in details if d["verdict"] == "FN"]
    if fn_list:
        print(f"\nFalse Negatives ({len(fn_list)}):")
        for d in fn_list:
            print(f"  {d['name']} ({d['category']}): {d['project']} — {GROUND_TRUTH[d['name']]['description']}")

    # False positives
    fp_list = [d for d in details if d["verdict"] == "FP"]
    if fp_list:
        print(f"\nFalse Positives ({len(fp_list)}):")
        for d in fp_list:
            print(f"  {d['name']} ({d['category']}): {d['project']} — bugs: {d['bugs_found']}")

    result_data = {
        "benchmark": "realworld_oss",
        "total_functions": len(REALWORLD_FUNCTIONS),
        "buggy": sum(1 for v in GROUND_TRUTH.values() if v["buggy"]),
        "safe": sum(1 for v in GROUND_TRUTH.values() if not v["buggy"]),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_time_ms": total_time,
        "per_category": {k: dict(v) for k, v in cat_results.items()},
        "per_project": {k: dict(v) for k, v in proj_results.items()},
        "details": details,
    }

    output_path = Path(__file__).parent / "results" / "E12_realworld_benchmark.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return result_data


if __name__ == "__main__":
    run_realworld_benchmark()
