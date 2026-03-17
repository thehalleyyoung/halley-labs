#!/usr/bin/env python3
"""
Real NegSynth Benchmark Runner

Invokes the compiled negsyn binary on its built-in benchmark suites
(slicer, merge, extract, encode, concretize, e2e) for both TLS and SSH
protocols. Captures actual timing, throughput, and memory data.

No simulated tools. Every number comes from running `negsyn benchmark`.
"""

import json
import subprocess
import sys
import os
import statistics
import time
from pathlib import Path

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"
NEGSYN_BIN = IMPL_DIR / "target" / "release" / "negsyn"
RESULTS_FILE = Path(__file__).resolve().parent / "real_benchmark_results.json"

PROTOCOLS = ["tls", "ssh"]
SUITES = ["slicer", "merge", "extract", "encode", "concretize", "e2e"]
ITERATIONS = 20
WARMUP = 3
OUTER_RUNS = 5  # repeat the full benchmark N times for statistical stability


def build_release():
    """Build NegSynth in release mode."""
    print("[build] cargo build --release ...")
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "negsyn"],
        cwd=str(IMPL_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[build] FAILED:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("[build] OK")


def run_benchmark(protocol: str, suite: str, iterations: int, warmup: int) -> dict:
    """Run a single benchmark invocation and return parsed JSON."""
    cmd = [
        str(NEGSYN_BIN),
        "benchmark", suite,
        "--format", "json",
        "--protocol", protocol,
        "-i", str(iterations),
        "--warmup", str(warmup),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"negsyn benchmark failed: {result.stderr}")
    return json.loads(result.stdout)


def aggregate_outer_runs(runs: list[dict]) -> dict:
    """Aggregate statistics across multiple outer runs of the same benchmark."""
    # Each run has the same structure; combine the 'results' arrays
    aggregated = {}
    for run in runs:
        for bench in run["suite"]["results"]:
            name = bench["name"]
            if name not in aggregated:
                aggregated[name] = {
                    "name": name,
                    "all_mean_ms": [],
                    "all_median_ms": [],
                    "all_min_ms": [],
                    "all_max_ms": [],
                    "throughput": bench["throughput"],
                    "memory_bytes": bench["memory_bytes"],
                }
            aggregated[name]["all_mean_ms"].append(bench["mean_ms"])
            aggregated[name]["all_median_ms"].append(bench["median_ms"])
            aggregated[name]["all_min_ms"].append(bench["min_ms"])
            aggregated[name]["all_max_ms"].append(bench["max_ms"])

    final = []
    for name, data in aggregated.items():
        means = data["all_mean_ms"]
        final.append({
            "name": name,
            "outer_runs": len(means),
            "inner_iterations": ITERATIONS,
            "mean_ms": statistics.mean(means),
            "median_ms": statistics.median(data["all_median_ms"]),
            "min_ms": min(data["all_min_ms"]),
            "max_ms": max(data["all_max_ms"]),
            "stddev_across_runs_ms": statistics.stdev(means) if len(means) > 1 else 0.0,
            "throughput": data["throughput"],
            "memory_bytes": data["memory_bytes"],
        })
    return final


def main():
    build_release()

    results = {
        "tool": "negsyn",
        "version": "0.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "methodology": (
            "Each benchmark invokes the compiled negsyn binary "
            f"({ITERATIONS} inner iterations, {WARMUP} warmup, "
            f"{OUTER_RUNS} outer runs). All numbers come from real execution."
        ),
        "inner_iterations": ITERATIONS,
        "warmup": WARMUP,
        "outer_runs": OUTER_RUNS,
        "protocols": {},
    }

    for proto in PROTOCOLS:
        print(f"\n{'='*60}")
        print(f"  Protocol: {proto.upper()}")
        print(f"{'='*60}")

        # Full-suite benchmarks
        outer_all = []
        for r in range(OUTER_RUNS):
            print(f"  [run {r+1}/{OUTER_RUNS}] negsyn benchmark all --protocol {proto} ...")
            data = run_benchmark(proto, "all", ITERATIONS, WARMUP)
            outer_all.append(data)

        aggregated = aggregate_outer_runs(outer_all)

        # Compute total e2e time
        e2e_entry = next((b for b in aggregated if b["name"] == "e2e"), None)
        total_e2e_ms = e2e_entry["mean_ms"] if e2e_entry else None

        results["protocols"][proto] = {
            "benchmarks": aggregated,
            "total_e2e_mean_ms": total_e2e_ms,
            "environment": outer_all[0]["suite"]["environment"],
        }

        # Print summary
        print(f"\n  {proto.upper()} Results:")
        print(f"  {'Phase':<15} {'Mean (ms)':>12} {'Median (ms)':>12} {'StdDev':>10}")
        print(f"  {'-'*49}")
        for b in aggregated:
            print(
                f"  {b['name']:<15} {b['mean_ms']:>12.4f} "
                f"{b['median_ms']:>12.4f} {b['stddev_across_runs_ms']:>10.4f}"
            )

    # Write results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
