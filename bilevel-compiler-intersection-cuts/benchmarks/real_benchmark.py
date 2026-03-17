#!/usr/bin/env python3
"""
Real benchmark for BiCut bilevel optimization compiler.

Exercises the actual Rust code end-to-end:
  1. Structural analysis
  2. Compilation (KKT, StrongDuality, ValueFunction reformulations)
  3. LP relaxation solving
  4. Branch-and-cut solving
  5. Cut generation and bound improvement

Tests 10 instances across 3 problem families:
  - Knapsack interdiction (3 sizes)
  - Network interdiction (3 sizes)
  - Toll pricing / network design (4 sizes)

Reports: reformulation time, model size, LP solve time, B&C solve time,
         gap closure, bound improvement, and comparison with/without cuts.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMPL_DIR = PROJECT_ROOT / "implementation"
BENCH_OUTPUT = PROJECT_ROOT / "benchmarks" / "benchmark_output"


def run_cargo_build():
    """Build the workspace in release mode."""
    print("=" * 70)
    print("  Building BiCut workspace (release)...")
    print("=" * 70)
    result = subprocess.run(
        ["cargo", "build", "--release", "--example", "real_benchmark_runner"],
        cwd=str(IMPL_DIR),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        return False
    print("  Build OK")
    return True


def run_benchmark_binary():
    """Run the Rust benchmark runner and capture JSON output."""
    binary = IMPL_DIR / "target" / "release" / "examples" / "real_benchmark_runner"
    if not binary.exists():
        print(f"Binary not found at {binary}")
        return None

    print("\n" + "=" * 70)
    print("  Running benchmark...")
    print("=" * 70)

    result = subprocess.run(
        [str(binary)],
        cwd=str(IMPL_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"Benchmark runner exited with code {result.returncode}")
        return None

    # Parse JSON from last line block
    lines = result.stdout.strip().split("\n")
    json_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('{"benchmark_results"'):
            json_start = i
            break
    if json_start is not None:
        json_text = "\n".join(lines[json_start:])
        return json.loads(json_text)
    return None


def format_results_table(results):
    """Format benchmark results as a readable table."""
    if not results or "benchmark_results" not in results:
        return "No results available."

    rows = results["benchmark_results"]
    header = (
        f"{'Instance':<30} {'Reform':>8} {'Vars':>6} {'Cons':>6} "
        f"{'Compile':>10} {'LP_Solve':>10} {'BC_Solve':>10} "
        f"{'Gap%':>8} {'Nodes':>8} {'Cuts':>6} {'Status':>10}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in rows:
        line = (
            f"{r['instance']:<30} {r['reformulation']:>8} "
            f"{r['num_vars']:>6} {r['num_constraints']:>6} "
            f"{r['compile_time_ms']:>9.1f}ms "
            f"{r['lp_solve_time_ms']:>9.2f}ms "
            f"{r['bc_solve_time_ms']:>9.1f}ms "
            f"{r['gap_pct']:>7.2f}% "
            f"{r['nodes']:>8} {r['cuts_added']:>6} "
            f"{r['status']:>10}"
        )
        lines.append(line)

    lines.append(sep)

    # Summary stats
    if rows:
        compile_times = [r["compile_time_ms"] for r in rows]
        lp_times = [r["lp_solve_time_ms"] for r in rows]
        bc_times = [r["bc_solve_time_ms"] for r in rows]
        gaps = [r["gap_pct"] for r in rows if r["gap_pct"] < 1e6]

        lines.append(f"\nSummary ({len(rows)} instances):")
        lines.append(f"  Compile time: mean={sum(compile_times)/len(compile_times):.1f}ms, "
                     f"max={max(compile_times):.1f}ms")
        lines.append(f"  LP solve:     mean={sum(lp_times)/len(lp_times):.2f}ms, "
                     f"max={max(lp_times):.2f}ms")
        lines.append(f"  B&C solve:    mean={sum(bc_times)/len(bc_times):.1f}ms, "
                     f"max={max(bc_times):.1f}ms")
        if gaps:
            lines.append(f"  Final gap:    mean={sum(gaps)/len(gaps):.2f}%, "
                         f"min={min(gaps):.2f}%, max={max(gaps):.2f}%")

    return "\n".join(lines)


def format_latex_table(results):
    """Format results as a LaTeX table for tool_paper.tex."""
    if not results or "benchmark_results" not in results:
        return ""

    rows = results["benchmark_results"]
    lines = []
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Instance & Reform. & Vars & Cons & Compile & LP & B\&C & Gap\% & Nodes \\"
    )
    lines.append(r"\midrule")

    for r in rows:
        name = r["instance"].replace("_", r"\_")
        lines.append(
            f"  {name} & {r['reformulation']} & {r['num_vars']} & "
            f"{r['num_constraints']} & {r['compile_time_ms']:.1f} & "
            f"{r['lp_solve_time_ms']:.1f} & {r['bc_solve_time_ms']:.1f} & "
            f"{r['gap_pct']:.2f} & {r['nodes']} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    os.makedirs(str(BENCH_OUTPUT), exist_ok=True)

    # Build
    if not run_cargo_build():
        print("\nFalling back: running cargo bench instead...")
        subprocess.run(["cargo", "bench"], cwd=str(IMPL_DIR), timeout=300)
        return

    # Run
    results = run_benchmark_binary()

    if results:
        # Console output
        table = format_results_table(results)
        print("\n" + table)

        # Save JSON
        out_json = BENCH_OUTPUT / "real_benchmark_results.json"
        with open(str(out_json), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_json}")

        # Save LaTeX fragment
        latex = format_latex_table(results)
        out_tex = BENCH_OUTPUT / "benchmark_table.tex"
        with open(str(out_tex), "w") as f:
            f.write(latex)
        print(f"LaTeX table saved to {out_tex}")
    else:
        print("\nNo structured results captured. Check output above.")


if __name__ == "__main__":
    main()
