#!/usr/bin/env python3
"""Timing benchmarks for litmus-cli: verify, compress, and benchmark commands."""

import csv
import os
import statistics
import subprocess
import time

BINARY = "./target/release/litmus-cli"
N_RUNS = 10
OUTPUT_DIR = "benchmark_results"
MODELS = ["SC", "TSO", "PSO", "ARM", "RISC-V"]

# Commands to benchmark: (label, args)
COMMANDS = []

# verify: test × model combinations specified in the task
for test in ["sb", "mp", "lb", "iriw", "sb4"]:
    for model in ["SC", "TSO"]:
        COMMANDS.append((f"verify {test} {model}", [BINARY, "verify", "-t", test, "-m", model]))

# compress
for test in ["sb", "mp", "lb", "iriw", "sb4"]:
    COMMANDS.append((f"compress {test}", [BINARY, "compress", "-t", test]))

# benchmark
COMMANDS.append(("benchmark", [BINARY, "benchmark"]))


def time_command(args, n=N_RUNS):
    """Run a command n times and return list of elapsed times in ms."""
    times = []
    for _ in range(n):
        start = time.time()
        subprocess.run(args, capture_output=True)
        elapsed_ms = (time.time() - start) * 1000.0
        times.append(elapsed_ms)
    return times


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Part 1: timing_stats.csv ---
    timing_rows = []
    for label, args in COMMANDS:
        print(f"Benchmarking: {label} ({N_RUNS} runs)...")
        times = time_command(args)
        row = {
            "command": label,
            "mean_time_ms": f"{statistics.mean(times):.3f}",
            "std_time_ms": f"{statistics.stdev(times):.3f}" if len(times) > 1 else "0.000",
            "min_time_ms": f"{min(times):.3f}",
            "max_time_ms": f"{max(times):.3f}",
            "n_runs": len(times),
        }
        timing_rows.append(row)
        print(f"  mean={row['mean_time_ms']}ms  std={row['std_time_ms']}ms")

    timing_path = os.path.join(OUTPUT_DIR, "timing_stats.csv")
    with open(timing_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["command", "mean_time_ms", "std_time_ms", "min_time_ms", "max_time_ms", "n_runs"])
        writer.writeheader()
        writer.writerows(timing_rows)
    print(f"\nWrote {timing_path}")

    # --- Part 2: cross_model_timing.csv ---
    # For each test, measure: single-model verify time vs all-5-model verify time.
    # Symmetry detection (compress) is done once and amortized across models.
    tests = ["sb", "mp", "lb", "iriw", "sb4"]
    cross_rows = []

    for test in tests:
        print(f"\nCross-model timing for {test}...")

        # Time compress (symmetry detection) once
        compress_times = time_command([BINARY, "compress", "-t", test])
        compress_mean = statistics.mean(compress_times)

        # Time single-model verify
        single_times = time_command([BINARY, "verify", "-t", test, "-m", "SC"])
        single_mean = statistics.mean(single_times)

        # Time all 5 models sequentially
        all_times = []
        for _ in range(N_RUNS):
            start = time.time()
            for model in MODELS:
                subprocess.run([BINARY, "verify", "-t", test, "-m", model], capture_output=True)
            elapsed_ms = (time.time() - start) * 1000.0
            all_times.append(elapsed_ms)
        all_mean = statistics.mean(all_times)

        # Amortization: symmetry detection cost / number of models
        n_models = len(MODELS)
        amortization_factor = compress_mean / n_models if compress_mean > 0 else 0.0

        cross_rows.append({
            "test": test,
            "n_models": n_models,
            "single_model_total_ms": f"{single_mean:.3f}",
            "all_models_ms": f"{all_mean:.3f}",
            "amortization_factor": f"{amortization_factor:.3f}",
        })
        print(f"  single={single_mean:.3f}ms  all_5={all_mean:.3f}ms  amort={amortization_factor:.3f}ms")

    cross_path = os.path.join(OUTPUT_DIR, "cross_model_timing.csv")
    with open(cross_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test", "n_models", "single_model_total_ms", "all_models_ms", "amortization_factor"])
        writer.writeheader()
        writer.writerows(cross_rows)
    print(f"\nWrote {cross_path}")
    print("Done.")


if __name__ == "__main__":
    main()
