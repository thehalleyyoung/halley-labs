#!/usr/bin/env python3
"""
run_all.py — Run all experiments and produce a summary report.

Usage:
    python experiments/run_all.py              # Run all 5 experiments
    python experiments/run_all.py --quick      # Quick mode (fewer cases)
    python experiments/run_all.py --exp 1 3    # Run only experiments 1 and 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure experiment modules and usability_oracle are importable
_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_root / "implementation"))

RESULTS_DIR = _here / "results"


def main():
    parser = argparse.ArgumentParser(description="Run usability oracle experiments")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer cases")
    parser.add_argument("--exp", nargs="*", type=int, help="Run specific experiments (1-5)")
    args = parser.parse_args()

    experiments = {
        1: ("Regression Detection Accuracy", "exp1_regression_detection"),
        2: ("Rank Correlation", "exp2_rank_correlation"),
        3: ("Scalability", "exp3_scalability"),
        4: ("Bottleneck Classification", "exp4_bottleneck_classification"),
        5: ("Ablation Study", "exp5_ablation"),
    }

    to_run = args.exp if args.exp else list(experiments.keys())

    print("╔" + "═" * 70 + "╗")
    print("║  Bounded-Rational Usability Oracle — Full Experiment Suite          ║")
    print("╚" + "═" * 70 + "╝")
    print(f"\n  Running experiments: {to_run}")
    if args.quick:
        print("  Mode: QUICK (reduced case counts)")
    print()

    overall_results = {}
    total_t0 = time.perf_counter()

    for exp_id in to_run:
        name, module_name = experiments[exp_id]
        print(f"\n{'━' * 72}")
        print(f"  [{exp_id}/5] {name}")
        print(f"{'━' * 72}")

        try:
            mod = __import__(module_name)
            result = mod.run_experiment()
            overall_results[f"exp{exp_id}"] = {"status": "success", "data": result}
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            overall_results[f"exp{exp_id}"] = {"status": "error", "error": str(e)}

    total_elapsed = time.perf_counter() - total_t0

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  FINAL SUMMARY")
    print("═" * 72)
    print(f"\n  Total wall-clock time: {total_elapsed:.1f}s")
    for exp_id in to_run:
        name, _ = experiments[exp_id]
        status = overall_results.get(f"exp{exp_id}", {}).get("status", "unknown")
        icon = "✓" if status == "success" else "✗"
        print(f"  {icon} Experiment {exp_id}: {name} — {status}")

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "total_time_sec": round(total_elapsed, 2),
                "experiments": overall_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
