"""
Real-world PyTorch benchmark evaluation for TensorGuard.

Evaluates TensorGuard on 56 benchmarks drawn from real-world PyTorch patterns
(ResNet, BERT, GPT, U-Net, DCGAN, ViT, etc.) to establish external validity.
Reports both coverage (fraction of models within the supported fragment) and
accuracy (F1 on covered models).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_checker import verify_model
from experiments.benchmarks.realworld_pytorch_benchmark import (
    REALWORLD_PYTORCH_BENCHMARKS,
    get_benchmark_summary,
)


def run_realworld_evaluation():
    """Run TensorGuard on all real-world PyTorch benchmarks."""
    results = {}
    tp = fp = tn = fn = 0
    errors = []
    parse_failures = []
    total_time_ms = 0.0
    custom_op_warnings_total = 0
    coverage_ratios = []

    print(f"Running TensorGuard on {len(REALWORLD_PYTORCH_BENCHMARKS)} "
          f"real-world PyTorch benchmarks...")
    print("=" * 70)

    for name, bench in sorted(REALWORLD_PYTORCH_BENCHMARKS.items()):
        source = bench["source"]
        input_shapes = bench["input_shapes"]
        is_buggy = bench["is_buggy"]
        category = bench["category"]

        t0 = time.monotonic()
        try:
            result = verify_model(source, input_shapes=input_shapes)
        except Exception as e:
            parse_failures.append({
                "name": name,
                "error": str(e),
                "category": category,
                "is_buggy": is_buggy,
            })
            print(f"  PARSE_FAIL {name}: {e}")
            continue
        elapsed = (time.monotonic() - t0) * 1000
        total_time_ms += elapsed

        detected_bug = not result.safe
        n_warnings = len(result.warnings) if result.warnings else 0
        custom_op_warnings_total += n_warnings
        cov = result.coverage
        if cov:
            coverage_ratios.append(cov.coverage_ratio)

        if is_buggy and detected_bug:
            verdict = "TP"
            tp += 1
        elif is_buggy and not detected_bug:
            verdict = "FN"
            fn += 1
        elif not is_buggy and detected_bug:
            verdict = "FP"
            fp += 1
        else:
            verdict = "TN"
            tn += 1

        ok = "✓" if verdict in ("TP", "TN") else "✗"
        warn_str = f" [⚠ {n_warnings} custom ops]" if n_warnings > 0 else ""
        print(f"  {ok} {verdict} {name} ({category}) "
              f"[{elapsed:.0f}ms]{warn_str}")

        results[name] = {
            "verdict": verdict,
            "detected_bug": detected_bug,
            "is_buggy": is_buggy,
            "category": category,
            "time_ms": round(elapsed, 1),
            "warnings": n_warnings,
            "coverage_ratio": cov.coverage_ratio if cov else None,
            "source_description": bench.get("source_description", ""),
        }

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    avg_coverage = (sum(coverage_ratios) / len(coverage_ratios)
                    if coverage_ratios else 0.0)
    full_coverage_count = sum(1 for r in coverage_ratios if r >= 1.0)

    # Per-category breakdown
    categories = {}
    for name, r in results.items():
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        categories[cat][r["verdict"].lower()] += 1

    cat_metrics = {}
    for cat, counts in categories.items():
        cat_p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 1.0
        cat_r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 1.0
        cat_f1 = 2 * cat_p * cat_r / (cat_p + cat_r) if (cat_p + cat_r) > 0 else 0.0
        cat_metrics[cat] = {
            "precision": round(cat_p, 3),
            "recall": round(cat_r, 3),
            "f1": round(cat_f1, 3),
            "tp": counts["tp"],
            "fp": counts["fp"],
            "tn": counts["tn"],
            "fn": counts["fn"],
        }

    print("\n" + "=" * 70)
    print("REAL-WORLD PYTORCH BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\nBenchmarks evaluated: {len(results)}")
    print(f"Parse failures: {len(parse_failures)}")
    print(f"\nAggregate metrics:")
    print(f"  F1:        {f1:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"\nCoverage:")
    print(f"  Avg operation coverage: {avg_coverage:.1%}")
    print(f"  Models with 100% coverage: {full_coverage_count}/{len(coverage_ratios)}")
    print(f"  Total custom op warnings: {custom_op_warnings_total}")
    print(f"\nTotal time: {total_time_ms:.0f}ms "
          f"({total_time_ms/len(results):.0f}ms avg)")

    print(f"\nPer-category breakdown:")
    for cat, m in sorted(cat_metrics.items()):
        print(f"  {cat:20s}: F1={m['f1']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  "
              f"(TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']})")

    if parse_failures:
        print(f"\nParse failures ({len(parse_failures)}):")
        for pf in parse_failures:
            print(f"  {pf['name']}: {pf['error'][:80]}")

    # Save results
    output = {
        "benchmark_suite": "realworld_pytorch",
        "n_benchmarks": len(REALWORLD_PYTORCH_BENCHMARKS),
        "n_evaluated": len(results),
        "n_parse_failures": len(parse_failures),
        "aggregate_metrics": {
            "f1": round(f1, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "accuracy": round(accuracy, 3),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "coverage": {
            "avg_operation_coverage": round(avg_coverage, 3),
            "full_coverage_count": full_coverage_count,
            "total_models": len(coverage_ratios),
            "custom_op_warnings": custom_op_warnings_total,
        },
        "per_category": cat_metrics,
        "total_time_ms": round(total_time_ms, 1),
        "per_benchmark": results,
        "parse_failures": parse_failures,
    }

    output_path = Path(__file__).resolve().parent / "realworld_pytorch_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    run_realworld_evaluation()
