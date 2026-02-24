"""
Comprehensive head-to-head comparison experiment.

Runs GuardHarvest, mypy, pyright, and pylint on:
1. Author benchmark (97 functions)
2. External benchmark (39 functions)
3. CVE benchmark (150 functions)
Then computes precision/recall/F1 for each tool and saves results.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.real_analyzer import analyze_source, FlowSensitiveAnalyzer


def run_tool_on_file(tool_name: str, filepath: str, timeout: int = 60) -> dict:
    """Run an external tool and parse its output for diagnostics."""
    diagnostics = []
    try:
        if tool_name == "mypy":
            result = subprocess.run(
                ["mypy", "--no-error-summary", "--show-column-numbers",
                 "--no-incremental", "--cache-dir=/dev/null", filepath],
                capture_output=True, text=True, timeout=timeout
            )
            for line in result.stdout.splitlines():
                if ": error:" in line or ": note:" in line:
                    diagnostics.append(line)

        elif tool_name == "pyright":
            result = subprocess.run(
                ["pyright", "--outputjson", filepath],
                capture_output=True, text=True, timeout=timeout
            )
            try:
                data = json.loads(result.stdout)
                for diag in data.get("generalDiagnostics", []):
                    if diag.get("severity") in ("error", "warning"):
                        diagnostics.append({
                            "line": diag.get("range", {}).get("start", {}).get("line", 0),
                            "message": diag.get("message", ""),
                            "severity": diag.get("severity", ""),
                            "rule": diag.get("rule", ""),
                        })
            except json.JSONDecodeError:
                pass

        elif tool_name == "pylint":
            result = subprocess.run(
                ["pylint", "--output-format=json", "--disable=all",
                 "--enable=E0602,E0611,E1101,E1120,E1121,E1123,E1124,E1125,"
                 "E1126,E1127,E1128,E1129,E1130,E1131,E1132,E1133,E1134,"
                 "E1135,E1136,E1137,E1138,E1139,E1140,E1141,W0104,W0106,"
                 "W0199,W0611,W0612,W0613",
                 filepath],
                capture_output=True, text=True, timeout=timeout
            )
            try:
                msgs = json.loads(result.stdout)
                for msg in msgs:
                    diagnostics.append({
                        "line": msg.get("line", 0),
                        "message": msg.get("message", ""),
                        "symbol": msg.get("symbol", ""),
                        "type": msg.get("type", ""),
                    })
            except json.JSONDecodeError:
                pass

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        pass

    return {"tool": tool_name, "diagnostics": diagnostics, "count": len(diagnostics)}


def run_guardharvest_on_functions(source: str, filepath: str) -> dict:
    """Run GuardHarvest and return per-function bug detection results."""
    result = analyze_source(source, filepath)
    func_bugs = {}
    for fr in result.function_results:
        func_bugs[fr.name] = {
            "detected": len(fr.bugs) > 0,
            "bugs": [{"category": b.category.name, "message": b.message,
                       "line": b.line} for b in fr.bugs],
            "guards": fr.guards_harvested,
            "predicates": fr.predicates_inferred,
        }
    return func_bugs


def detect_function_ranges(source: str) -> dict:
    """Map function names to line ranges."""
    tree = ast.parse(source)
    ranges = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, 'end_lineno', node.lineno + 10)
            ranges[node.name] = (node.lineno, end)
    return ranges


def tool_detects_in_range(diagnostics: list, start_line: int, end_line: int) -> bool:
    """Check if any diagnostic falls within the function's line range."""
    for d in diagnostics:
        if isinstance(d, dict):
            line = d.get("line", 0)
            if start_line <= line <= end_line:
                return True
        elif isinstance(d, str):
            # Parse "file.py:line:col: error: msg"
            parts = d.split(":")
            if len(parts) >= 2:
                try:
                    line = int(parts[1])
                    if start_line <= line <= end_line:
                        return True
                except ValueError:
                    pass
    return False


def evaluate_benchmark(benchmark_name: str, source: str, ground_truth: dict,
                       filepath: str) -> dict:
    """Run all tools on a benchmark and compute metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Functions: {len(ground_truth)}")
    print(f"{'='*60}")

    # Get function line ranges
    func_ranges = detect_function_ranges(source)

    # Run GuardHarvest
    print("  Running GuardHarvest...")
    t0 = time.perf_counter()
    gh_results = run_guardharvest_on_functions(source, filepath)
    gh_time = (time.perf_counter() - t0) * 1000

    # Run external tools
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source)
        tmp_path = f.name

    try:
        tools_results = {}
        for tool in ["mypy", "pyright", "pylint"]:
            print(f"  Running {tool}...")
            t0 = time.perf_counter()
            result = run_tool_on_file(tool, tmp_path)
            elapsed = (time.perf_counter() - t0) * 1000
            tools_results[tool] = {
                "diagnostics": result["diagnostics"],
                "time_ms": round(elapsed, 1),
            }
    finally:
        os.unlink(tmp_path)

    # Compute metrics for each tool
    all_metrics = {}

    # GuardHarvest metrics
    tp = fp = fn = tn = 0
    per_func = []
    for func_name, truth in ground_truth.items():
        has_bug = truth["has_bug"]
        detected = gh_results.get(func_name, {}).get("detected", False)

        if has_bug and detected: tp += 1; v = "TP"
        elif not has_bug and detected: fp += 1; v = "FP"
        elif has_bug and not detected: fn += 1; v = "FN"
        else: tn += 1; v = "TN"

        per_func.append({"function": func_name, "verdict": v,
                         "category": truth["category"],
                         "has_bug": has_bug, "detected": detected})

    p = tp/(tp+fp) if (tp+fp) > 0 else 0
    r = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    all_metrics["guardharvest"] = {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(p, 4), "recall": round(r, 4),
        "F1": round(f1, 4), "time_ms": round(gh_time, 1),
        "per_function": per_func,
    }
    print(f"  GuardHarvest: P={p:.1%} R={r:.1%} F1={f1:.3f} ({gh_time:.0f}ms)")

    # External tool metrics
    for tool_name, tool_data in tools_results.items():
        tp = fp = fn = tn = 0
        per_func = []
        for func_name, truth in ground_truth.items():
            has_bug = truth["has_bug"]
            rng = func_ranges.get(func_name, (0, 0))
            detected = tool_detects_in_range(tool_data["diagnostics"],
                                              rng[0], rng[1])

            if has_bug and detected: tp += 1; v = "TP"
            elif not has_bug and detected: fp += 1; v = "FP"
            elif has_bug and not detected: fn += 1; v = "FN"
            else: tn += 1; v = "TN"

            per_func.append({"function": func_name, "verdict": v,
                             "category": truth["category"],
                             "has_bug": has_bug, "detected": detected})

        p = tp/(tp+fp) if (tp+fp) > 0 else 0
        r = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        all_metrics[tool_name] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(p, 4), "recall": round(r, 4),
            "F1": round(f1, 4), "time_ms": tool_data["time_ms"],
            "per_function": per_func,
        }
        print(f"  {tool_name}: P={p:.1%} R={r:.1%} F1={f1:.3f} ({tool_data['time_ms']:.0f}ms)")

    # Category breakdown for GuardHarvest
    categories = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for pf in all_metrics["guardharvest"]["per_function"]:
        cat = pf["category"]
        v = pf["verdict"]
        categories[cat][v.lower()] += 1

    cat_metrics = {}
    for cat, counts in categories.items():
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        p = tp/(tp+fp) if (tp+fp) > 0 else (1.0 if fn == 0 else 0)
        r = tp/(tp+fn) if (tp+fn) > 0 else (1.0 if tp > 0 else 0)
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        cat_metrics[cat] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                            "precision": round(p, 3), "recall": round(r, 3),
                            "f1": round(f1, 3)}
        print(f"    {cat}: P={p:.1%} R={r:.1%} F1={f1:.3f}")

    return {
        "benchmark": benchmark_name,
        "total_functions": len(ground_truth),
        "buggy_functions": sum(1 for t in ground_truth.values() if t["has_bug"]),
        "safe_functions": sum(1 for t in ground_truth.values() if not t["has_bug"]),
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "per_function"}
                    for k, v in all_metrics.items()},
        "category_breakdown": cat_metrics,
        "per_function": {k: v["per_function"] for k, v in all_metrics.items()},
    }


def main():
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    # Benchmark 1: Author benchmark (97 functions)
    bench1_path = Path(__file__).parent / "benchmarks" / "benchmark_suite.py"
    bench1_source = bench1_path.read_text()
    sys.path.insert(0, str(bench1_path.parent))
    from benchmarks.benchmark_suite import GROUND_TRUTH as GT1
    r1 = evaluate_benchmark("author_benchmark_97fn", bench1_source, GT1, str(bench1_path))
    all_results["author_benchmark"] = r1

    # Benchmark 2: External benchmark (39 functions)
    bench2_path = Path(__file__).parent / "benchmarks" / "external_benchmark.py"
    bench2_source = bench2_path.read_text()
    from benchmarks.external_benchmark import GROUND_TRUTH as GT2
    r2 = evaluate_benchmark("external_benchmark_39fn", bench2_source, GT2, str(bench2_path))
    all_results["external_benchmark"] = r2

    # Benchmark 3: CVE benchmark (150 functions)
    bench3_path = Path(__file__).parent / "benchmarks" / "cve_benchmark.py"
    bench3_source = bench3_path.read_text()
    from benchmarks.cve_benchmark import GROUND_TRUTH as GT3
    r3 = evaluate_benchmark("cve_benchmark_150fn", bench3_source, GT3, str(bench3_path))
    all_results["cve_benchmark"] = r3

    # Combined metrics across all benchmarks
    print("\n" + "=" * 60)
    print("COMBINED RESULTS ACROSS ALL BENCHMARKS")
    print("=" * 60)

    total_funcs = sum(r["total_functions"] for r in [r1, r2, r3])
    print(f"Total functions: {total_funcs}")

    for tool in ["guardharvest", "mypy", "pyright", "pylint"]:
        total_tp = sum(r["metrics"][tool]["TP"] for r in [r1, r2, r3])
        total_fp = sum(r["metrics"][tool]["FP"] for r in [r1, r2, r3])
        total_fn = sum(r["metrics"][tool]["FN"] for r in [r1, r2, r3])
        total_tn = sum(r["metrics"][tool]["TN"] for r in [r1, r2, r3])
        p = total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0
        r = total_tp/(total_tp+total_fn) if (total_tp+total_fn) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        total_time = sum(rr["metrics"][tool]["time_ms"] for rr in [r1, r2, r3])
        print(f"  {tool:15s}: P={p:.1%} R={r:.1%} F1={f1:.3f} "
              f"(TP={total_tp} FP={total_fp} FN={total_fn} TN={total_tn}) "
              f"[{total_time:.0f}ms]")
        all_results[f"combined_{tool}"] = {
            "TP": total_tp, "FP": total_fp, "FN": total_fn, "TN": total_tn,
            "precision": round(p, 4), "recall": round(r, 4), "F1": round(f1, 4),
            "total_time_ms": round(total_time, 1),
        }

    # Save results
    save_path = output_dir / "comprehensive_comparison.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return all_results


if __name__ == "__main__":
    results = main()
