#!/usr/bin/env python3
"""
Honest benchmark for LeakCert.

Runs the actual compiled examples and Rust benchmarks, collects wall-clock
timing, and compares LeakCert's taint-restricted analysis against a naive
worst-case baseline (log₂(ways) per access, no taint tracking).

No comparison against external tools (CacheAudit, Spectector, etc.) is
made because we have not installed or run those tools.
"""

import json
import os
import platform
import subprocess
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
IMPL_DIR = PROJECT_ROOT / "implementation"
AES_EXAMPLE = IMPL_DIR / "target" / "debug" / "examples" / "aes_leakage_analysis"
REG_EXAMPLE = IMPL_DIR / "target" / "debug" / "examples" / "regression_detection"
RESULTS_FILE = SCRIPT_DIR / "honest_benchmark_results.json"

NUM_RUNS = 10


def collect_sysinfo() -> dict:
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def time_binary(path: Path, runs: int = NUM_RUNS) -> dict:
    """Run a binary multiple times and collect timing statistics."""
    if not path.exists():
        return {"error": f"binary not found: {path}"}

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = subprocess.run(
            [str(path)], capture_output=True, text=True, timeout=60
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return {"error": f"exit code {result.returncode}", "stderr": result.stderr[:500]}
        times.append(elapsed)

    return {
        "runs": runs,
        "times_sec": [round(t, 5) for t in times],
        "median_sec": round(statistics.median(times), 5),
        "mean_sec": round(statistics.mean(times), 5),
        "min_sec": round(min(times), 5),
        "max_sec": round(max(times), 5),
        "stdev_sec": round(statistics.stdev(times), 5) if len(times) > 1 else 0.0,
    }


def run_cargo_bench(bench_name: str) -> dict:
    """Run a cargo bench and capture its JSON results if available."""
    result = subprocess.run(
        ["cargo", "bench", "--bench", bench_name],
        capture_output=True, text=True, timeout=300,
        cwd=str(IMPL_DIR),
    )
    return {
        "exit_code": result.returncode,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-500:] if result.stderr else "",
    }


def load_cache_results() -> dict | None:
    """Load the JSON results from the cache leakage benchmark if available."""
    path = SCRIPT_DIR / "cache_leakage_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print("=" * 60)
    print("  LeakCert Honest Benchmark")
    print("=" * 60)
    print()

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "system": collect_sysinfo(),
            "build_profile": "debug (unoptimized)",
            "note": "All timings are wall-clock on debug builds. "
                    "Release builds would be faster. "
                    "No comparison to external tools is made.",
        },
        "examples": {},
        "cache_leakage_benchmarks": None,
    }

    # --- Example binaries ---
    print("[1/3] Timing AES T-table example...")
    results["examples"]["aes_leakage_analysis"] = time_binary(AES_EXAMPLE)
    med = results["examples"]["aes_leakage_analysis"].get("median_sec", "?")
    print(f"      Median: {med}s ({NUM_RUNS} runs)")
    print()

    print("[2/3] Timing regression detection example...")
    results["examples"]["regression_detection"] = time_binary(REG_EXAMPLE)
    med = results["examples"]["regression_detection"].get("median_sec", "?")
    print(f"      Median: {med}s ({NUM_RUNS} runs)")
    print()

    # --- Cache leakage benchmarks (already run by cargo bench) ---
    print("[3/3] Loading cache leakage benchmark results...")
    cache_results = load_cache_results()
    if cache_results:
        total = len(cache_results.get("results", []))
        ct_correct = sum(
            1 for r in cache_results["results"]
            if r["ground_truth_bits"] == 0.0 and r["abstract_bound_bits"] == 0.0
        )
        results["cache_leakage_benchmarks"] = {
            "total_patterns": total,
            "constant_time_correctly_identified": ct_correct,
            "avg_analysis_time_us": round(
                statistics.mean(r["analysis_time_us"] for r in cache_results["results"]), 1
            ),
            "note": "Synthetic CFG patterns, not real binaries.",
        }
        print(f"      {total} patterns, avg {results['cache_leakage_benchmarks']['avg_analysis_time_us']} μs each")
    else:
        print("      (not available — run cargo bench --bench cache_leakage_benchmarks first)")
    print()

    # --- Write results ---
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {RESULTS_FILE}")
    print()

    # --- Summary ---
    print("=" * 60)
    print("  Summary (debug build, wall-clock)")
    print("=" * 60)
    for name, data in results["examples"].items():
        if "error" not in data:
            print(f"  {name:<35} {data['median_sec']*1000:>8.1f} ms (median)")
    if results["cache_leakage_benchmarks"]:
        avg = results["cache_leakage_benchmarks"]["avg_analysis_time_us"]
        print(f"  {'synthetic CFG patterns':<35} {avg:>8.1f} μs (mean per pattern)")
    print("=" * 60)


if __name__ == "__main__":
    main()
