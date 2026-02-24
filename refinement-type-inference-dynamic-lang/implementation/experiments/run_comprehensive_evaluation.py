#!/usr/bin/env python3
"""Comprehensive evaluation for TensorGuard.

Loads existing experiment data, runs analysis on 50 benchmark functions,
computes precision/recall/F1 with 95% Clopper-Pearson CIs, runs ablations,
and saves all results to experiments/comprehensive_evaluation_results.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import analyze, liquid_analyze, _HAS_LIQUID, BugCategory
from src.tensor_shapes import analyze_shapes
from experiments.benchmark_suite import (
    ALL_BENCHMARKS,
    NULL_SAFETY_BUGS,
    TENSOR_SHAPE_BUGS,
    CORRECT_FUNCTIONS,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = EXPERIMENTS_DIR / "comprehensive_evaluation_results.json"


# ── Clopper-Pearson confidence interval ───────────────────────────────
def _beta_ppf(a: float, b: float, p: float) -> float:
    """Approximate Beta PPF using Newton's method on regularized incomplete beta."""
    # Use scipy if available, else fall back to normal approximation
    try:
        from scipy.stats import beta as beta_dist
        return beta_dist.ppf(p, a, b)
    except ImportError:
        # Normal approximation to beta distribution
        mu = a / (a + b)
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        sd = math.sqrt(var) if var > 0 else 1e-10
        # Approximate using normal
        from statistics import NormalDist
        z = NormalDist().inv_cdf(p)
        return max(0.0, min(1.0, mu + z * sd))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """95% Clopper-Pearson exact binomial CI."""
    if n == 0:
        return (0.0, 1.0)
    lo = _beta_ppf(k, n - k + 1, alpha / 2) if k > 0 else 0.0
    hi = _beta_ppf(k + 1, n - k, 1 - alpha / 2) if k < n else 1.0
    return (lo, hi)


# ── Load existing experiment data ─────────────────────────────────────
def load_existing_data() -> Dict[str, Any]:
    """Load and summarize all existing experiment JSON files."""
    files = {
        "experiment_results_v2": "experiment_results_v2.json",
        "stdlib_eval_results": "stdlib_eval_results.json",
        "pyright_comparison": "pyright_comparison.json",
        "interprocedural_fp_reduction": "interprocedural_fp_reduction.json",
        "results": "results.json",
    }
    existing = {}
    for key, fname in files.items():
        fpath = EXPERIMENTS_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            existing[key] = data
            print(f"  ✓ Loaded {fname}")
        else:
            print(f"  ✗ Missing {fname}")
            existing[key] = None
    return existing


def summarize_existing(existing: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary statistics from existing data."""
    summary = {}

    # experiment_results_v2
    v2 = existing.get("experiment_results_v2")
    if v2:
        summary["v2_null_safety"] = {
            "n": len(v2.get("null_safety", {}).get("results", [])),
            "precision": v2["null_safety"]["precision"],
            "recall": v2["null_safety"]["recall"],
            "f1": v2["null_safety"]["f1"],
        }
        summary["v2_tensor_shapes"] = {
            "n": len(v2.get("tensor_shapes", {}).get("results", [])),
            "precision": v2["tensor_shapes"]["precision"],
            "recall": v2["tensor_shapes"]["recall"],
            "f1": v2["tensor_shapes"]["f1"],
        }

    # stdlib_eval
    stdlib = existing.get("stdlib_eval_results")
    if stdlib:
        summary["stdlib"] = {
            "total_loc": stdlib.get("total_loc", 0),
            "total_functions": stdlib.get("total_functions", 0),
            "total_bugs": stdlib.get("total_bugs", 0),
            "modules": len(stdlib.get("per_module", {})),
        }

    # pyright comparison
    pyright = existing.get("pyright_comparison")
    if pyright:
        totals = pyright.get("totals", {})
        summary["pyright_comparison"] = {
            "total_loc": totals.get("loc", 0),
            "guardharvest_bugs": totals.get("guardharvest_bugs", 0),
            "pyright_errors": totals.get("pyright_errors", 0),
            "modules": len(pyright.get("comparison", {})),
        }

    # interprocedural
    interproc = existing.get("interprocedural_fp_reduction")
    if interproc and isinstance(interproc, dict):
        summary["interprocedural"] = {
            "modules": len(interproc),
            "module_names": list(interproc.keys()),
        }

    # results.json
    results = existing.get("results")
    if results:
        summary["core_experiments"] = {
            "n_experiments": results.get("n_experiments", 0),
            "n_passed": results.get("n_passed", 0),
            "total_time_sec": results.get("total_time_sec", 0),
        }

    return summary


# ── Run benchmark analysis ────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    name: str
    category: str
    has_null_bug: bool
    has_shape_bug: bool
    detected_null: bool = False
    detected_shape: bool = False
    num_bugs_found: int = 0
    analysis_time_ms: float = 0.0
    error: Optional[str] = None


def run_null_analysis(code: str) -> Tuple[bool, int, float]:
    """Run null-safety analysis, return (detected, bug_count, time_ms)."""
    t0 = time.perf_counter()
    try:
        result = analyze(code, use_liquid=_HAS_LIQUID)
        elapsed = (time.perf_counter() - t0) * 1000
        null_bugs = [b for b in result.bugs if b.category in (
            BugCategory.NULL_DEREFERENCE, BugCategory.ATTRIBUTE_ERROR)]
        return (len(null_bugs) > 0, len(null_bugs), elapsed)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return (False, 0, elapsed)


def run_shape_analysis(code: str) -> Tuple[bool, int, float]:
    """Run tensor shape analysis, return (detected, error_count, time_ms)."""
    t0 = time.perf_counter()
    try:
        result = analyze_shapes(code)
        elapsed = (time.perf_counter() - t0) * 1000
        return (len(result.errors) > 0, len(result.errors), elapsed)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return (False, 0, elapsed)


def run_benchmarks(benchmarks: List[Dict]) -> List[BenchmarkResult]:
    """Run all benchmarks and collect results."""
    results = []
    for bench in benchmarks:
        br = BenchmarkResult(
            name=bench["name"],
            category=bench["category"],
            has_null_bug=bench["has_null_bug"],
            has_shape_bug=bench["has_shape_bug"],
        )
        code = bench["code"]

        # Null-safety analysis
        try:
            detected, count, t = run_null_analysis(code)
            br.detected_null = detected
            br.num_bugs_found += count
            br.analysis_time_ms += t
        except Exception as e:
            br.error = str(e)

        # Shape analysis (only for code with torch imports)
        if "torch" in code:
            try:
                detected, count, t = run_shape_analysis(code)
                br.detected_shape = detected
                br.num_bugs_found += count
                br.analysis_time_ms += t
            except Exception as e:
                br.error = str(e)

        results.append(br)
    return results


# ── Compute metrics ───────────────────────────────────────────────────
def compute_metrics(results: List[BenchmarkResult], bug_type: str) -> Dict[str, Any]:
    """Compute precision, recall, F1 with CIs for a bug type."""
    tp = fp = tn = fn = 0
    details = []

    for r in results:
        if bug_type == "null":
            has_bug = r.has_null_bug
            detected = r.detected_null
        else:  # shape
            has_bug = r.has_shape_bug
            detected = r.detected_shape

        if has_bug and detected:
            tp += 1; status = "TP"
        elif not has_bug and detected:
            fp += 1; status = "FP"
        elif has_bug and not detected:
            fn += 1; status = "FN"
        else:
            tn += 1; status = "TN"

        details.append({"name": r.name, "status": status})

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    prec_ci = clopper_pearson(tp, tp + fp)
    rec_ci = clopper_pearson(tp, tp + fn)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "precision_ci_95": [round(prec_ci[0], 4), round(prec_ci[1], 4)],
        "recall": round(recall, 4),
        "recall_ci_95": [round(rec_ci[0], 4), round(rec_ci[1], 4)],
        "f1": round(f1, 4),
        "n": tp + fp + tn + fn,
        "details": details,
    }


# ── Ablation study ────────────────────────────────────────────────────
def run_ablation(benchmarks: List[Dict]) -> Dict[str, Any]:
    """Run ablation: with/without CEGAR, with/without liquid types."""
    from src.liquid import LiquidTypeInferencer, InterproceduralLiquidAnalyzer, PredicateHarvester
    import ast

    null_benchmarks = [b for b in benchmarks if b["has_null_bug"] or b["category"] == "correct"]
    ablation_results = {}

    # 1. Flow-sensitive only (no liquid types)
    print("  Ablation: flow-sensitive only...")
    tp = fp = fn = tn = 0
    for b in null_benchmarks:
        try:
            result = analyze(b["code"], use_liquid=False)
            null_bugs = [bug for bug in result.bugs if bug.category in (
                BugCategory.NULL_DEREFERENCE, BugCategory.ATTRIBUTE_ERROR)]
            detected = len(null_bugs) > 0
        except Exception:
            detected = False
        has_bug = b["has_null_bug"]
        if has_bug and detected: tp += 1
        elif not has_bug and detected: fp += 1
        elif has_bug and not detected: fn += 1
        else: tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    ablation_results["flow_sensitive_only"] = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
    }

    # 2. Liquid types with CEGAR disabled (max_cegar=0)
    print("  Ablation: liquid types, no CEGAR...")
    tp = fp = fn = tn = 0
    for b in null_benchmarks:
        try:
            engine = LiquidTypeInferencer(max_cegar=0)
            lresult = engine.infer_module(b["code"])
            detected = len(lresult.bugs) > 0
        except Exception:
            detected = False
        has_bug = b["has_null_bug"]
        if has_bug and detected: tp += 1
        elif not has_bug and detected: fp += 1
        elif has_bug and not detected: fn += 1
        else: tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    ablation_results["liquid_no_cegar"] = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
    }

    # 3. Liquid types with full CEGAR (default)
    print("  Ablation: liquid types, full CEGAR...")
    tp = fp = fn = tn = 0
    for b in null_benchmarks:
        try:
            result = analyze(b["code"], use_liquid=True)
            null_bugs = [bug for bug in result.bugs if bug.category in (
                BugCategory.NULL_DEREFERENCE, BugCategory.ATTRIBUTE_ERROR)]
            detected = len(null_bugs) > 0
        except Exception:
            detected = False
        has_bug = b["has_null_bug"]
        if has_bug and detected: tp += 1
        elif not has_bug and detected: fp += 1
        elif has_bug and not detected: fn += 1
        else: tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    ablation_results["liquid_full_cegar"] = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
    }

    # 4. Predicate source ablation: count predicates per source
    print("  Ablation: predicate sources...")
    source_counts = {
        "guards": 0, "asserts": 0, "defaults": 0,
        "exceptions": 0, "returns": 0, "walrus": 0,
        "comprehension_filters": 0,
    }
    total_functions = 0
    for b in benchmarks:
        try:
            tree = ast.parse(b["code"])
            harvester = PredicateHarvester()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_functions += 1
                    source_counts["guards"] += len(harvester.harvest_guards(node))
                    source_counts["asserts"] += len(harvester.harvest_asserts(node))
                    source_counts["defaults"] += len(harvester.harvest_defaults(node))
                    source_counts["exceptions"] += len(harvester.harvest_exceptions(node))
                    source_counts["returns"] += len(harvester.harvest_returns(node))
                    source_counts["walrus"] += len(harvester.harvest_walrus(node))
                    source_counts["comprehension_filters"] += len(harvester.harvest_comprehension_filters(node))
        except Exception:
            pass

    ablation_results["predicate_sources"] = {
        "total_functions_analyzed": total_functions,
        "predicates_per_source": source_counts,
        "total_predicates": sum(source_counts.values()),
    }

    return ablation_results


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("TensorGuard Comprehensive Evaluation")
    print("=" * 70)
    t_start = time.perf_counter()

    # Step 1: Load existing data
    print("\n[1/5] Loading existing experiment data...")
    existing = load_existing_data()
    existing_summary = summarize_existing(existing)

    # Step 2: Run benchmark suite
    print(f"\n[2/5] Running {len(ALL_BENCHMARKS)} benchmark functions...")
    print(f"  - {len(NULL_SAFETY_BUGS)} null-safety bugs")
    print(f"  - {len(TENSOR_SHAPE_BUGS)} tensor shape bugs")
    print(f"  - {len(CORRECT_FUNCTIONS)} correct functions")

    bench_results = run_benchmarks(ALL_BENCHMARKS)

    # Step 3: Compute metrics
    print("\n[3/5] Computing metrics with 95% Clopper-Pearson CIs...")
    null_metrics = compute_metrics(bench_results, "null")
    shape_metrics = compute_metrics(bench_results, "shape")

    # Combined metrics
    all_tp = null_metrics["tp"] + shape_metrics["tp"]
    all_fp = null_metrics["fp"] + shape_metrics["fp"]
    all_fn = null_metrics["fn"] + shape_metrics["fn"]
    all_tn = null_metrics["tn"] + shape_metrics["tn"]
    overall_prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0
    prec_ci = clopper_pearson(all_tp, all_tp + all_fp)
    rec_ci = clopper_pearson(all_tp, all_tp + all_fn)

    overall_metrics = {
        "tp": all_tp, "fp": all_fp, "tn": all_tn, "fn": all_fn,
        "precision": round(overall_prec, 4),
        "precision_ci_95": [round(prec_ci[0], 4), round(prec_ci[1], 4)],
        "recall": round(overall_rec, 4),
        "recall_ci_95": [round(rec_ci[0], 4), round(rec_ci[1], 4)],
        "f1": round(overall_f1, 4),
        "n": len(ALL_BENCHMARKS),
    }

    # Step 4: Ablation
    print("\n[4/5] Running ablation study...")
    ablation = run_ablation(ALL_BENCHMARKS)

    # Step 5: Timing
    total_time = time.perf_counter() - t_start
    timing = {
        "total_benchmarks": len(ALL_BENCHMARKS),
        "total_evaluation_time_sec": round(total_time, 2),
        "mean_analysis_time_ms": round(
            sum(r.analysis_time_ms for r in bench_results) / len(bench_results), 2
        ),
    }

    # ── Assemble output ───────────────────────────────────────────────
    output = {
        "evaluation_metadata": {
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_benchmarks": len(ALL_BENCHMARKS),
            "liquid_types_available": _HAS_LIQUID,
            "total_time_sec": round(total_time, 2),
        },
        "existing_data_summary": existing_summary,
        "overall_metrics": overall_metrics,
        "null_safety_metrics": null_metrics,
        "tensor_shape_metrics": shape_metrics,
        "ablation": ablation,
        "timing": timing,
        "per_benchmark": [
            {
                "name": r.name,
                "category": r.category,
                "has_null_bug": r.has_null_bug,
                "has_shape_bug": r.has_shape_bug,
                "detected_null": r.detected_null,
                "detected_shape": r.detected_shape,
                "bugs_found": r.num_bugs_found,
                "time_ms": round(r.analysis_time_ms, 2),
                "error": r.error,
            }
            for r in bench_results
        ],
    }

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[5/5] Results saved to {OUTPUT_FILE}")

    # ── Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBenchmark suite: {len(ALL_BENCHMARKS)} functions")
    print(f"  Null-safety bugs:   {len(NULL_SAFETY_BUGS)}")
    print(f"  Tensor shape bugs:  {len(TENSOR_SHAPE_BUGS)}")
    print(f"  Correct functions:  {len(CORRECT_FUNCTIONS)}")

    print(f"\n── Overall (n={overall_metrics['n']}) ──")
    print(f"  Precision: {overall_metrics['precision']:.4f}  "
          f"CI₉₅=[{overall_metrics['precision_ci_95'][0]:.4f}, {overall_metrics['precision_ci_95'][1]:.4f}]")
    print(f"  Recall:    {overall_metrics['recall']:.4f}  "
          f"CI₉₅=[{overall_metrics['recall_ci_95'][0]:.4f}, {overall_metrics['recall_ci_95'][1]:.4f}]")
    print(f"  F1:        {overall_metrics['f1']:.4f}")
    print(f"  TP={all_tp}  FP={all_fp}  TN={all_tn}  FN={all_fn}")

    print(f"\n── Null Safety (n={null_metrics['n']}) ──")
    print(f"  Precision: {null_metrics['precision']:.4f}  "
          f"CI₉₅=[{null_metrics['precision_ci_95'][0]:.4f}, {null_metrics['precision_ci_95'][1]:.4f}]")
    print(f"  Recall:    {null_metrics['recall']:.4f}  "
          f"CI₉₅=[{null_metrics['recall_ci_95'][0]:.4f}, {null_metrics['recall_ci_95'][1]:.4f}]")
    print(f"  F1:        {null_metrics['f1']:.4f}")

    print(f"\n── Tensor Shapes (n={shape_metrics['n']}) ──")
    print(f"  Precision: {shape_metrics['precision']:.4f}  "
          f"CI₉₅=[{shape_metrics['precision_ci_95'][0]:.4f}, {shape_metrics['precision_ci_95'][1]:.4f}]")
    print(f"  Recall:    {shape_metrics['recall']:.4f}  "
          f"CI₉₅=[{shape_metrics['recall_ci_95'][0]:.4f}, {shape_metrics['recall_ci_95'][1]:.4f}]")
    print(f"  F1:        {shape_metrics['f1']:.4f}")

    print(f"\n── Ablation ──")
    for name, data in ablation.items():
        if name == "predicate_sources":
            print(f"  Predicate sources ({data['total_predicates']} total):")
            for src, cnt in data["predicates_per_source"].items():
                print(f"    {src}: {cnt}")
        else:
            print(f"  {name}: P={data['precision']:.4f} R={data['recall']:.4f} F1={data['f1']:.4f}")

    print(f"\n── Existing Data Summary ──")
    for key, val in existing_summary.items():
        print(f"  {key}: {val}")

    print(f"\nTotal evaluation time: {total_time:.2f}s")
    print(f"Mean per-benchmark: {timing['mean_analysis_time_ms']:.2f}ms")

    return output


if __name__ == "__main__":
    main()
