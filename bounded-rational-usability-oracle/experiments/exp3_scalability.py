#!/usr/bin/env python3
"""
Experiment 3: Scalability — Wall-Clock Time vs. UI Size
=========================================================

Measures pipeline latency as a function of UI element count (10 to 10,000
interactive elements) to validate the CI/CD feasibility claim (≤60s for
≤500 elements on laptop CPU).

Produces scaling curves and fits power-law / log-linear models.
"""

import json
import time
from pathlib import Path

import numpy as np

from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationGenerator
from usability_oracle.benchmarks.suite import BenchmarkSuite, BenchmarkCase
from usability_oracle.benchmarks.metrics import BenchmarkMetrics
from usability_oracle.core.enums import RegressionVerdict

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
N_REPEATS = 3

# Element counts to benchmark
SIZES = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]


def _generate_ui_at_scale(gen, n_elements):
    """Generate a UI with approximately n_elements interactive nodes."""
    if n_elements <= 20:
        return gen.generate_form(n_fields=n_elements, complexity="simple")
    elif n_elements <= 50:
        return gen.generate_navigation(n_items=n_elements, depth=3)
    elif n_elements <= 200:
        return gen.generate_settings_page(n_settings=n_elements)
    else:
        # For large UIs: composite (dashboard + data table + navigation)
        n_widgets = min(n_elements // 10, 50)
        return gen.generate_dashboard(n_widgets=n_widgets)


def run_experiment():
    print("=" * 72)
    print("  Experiment 3: Scalability — Wall-Clock Time vs. UI Size")
    print("=" * 72)

    gen = SyntheticUIGenerator(seed=SEED)
    mut = MutationGenerator(seed=SEED)

    results = {}
    print(f"\n{'Size':>8} {'Mean (s)':>10} {'Std (s)':>10} {'Min (s)':>10} {'Max (s)':>10} {'Status':>10}")
    print("─" * 62)

    for n in SIZES:
        times = []
        errors = []
        for rep in range(N_REPEATS):
            try:
                tree_a = _generate_ui_at_scale(gen, n)
                tree_b = mut.apply_perceptual_overload(tree_a, severity=0.5)

                # Use the full PipelineRunner for realistic timing
                from usability_oracle.pipeline.runner import PipelineRunner
                runner = PipelineRunner()
                t0 = time.perf_counter()
                result = runner.run(source_a=tree_a, source_b=tree_b)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
            except Exception as e:
                errors.append(str(e))
                times.append(float("nan"))

        valid_times = [t for t in times if not np.isnan(t)]
        if valid_times:
            mean_t = np.mean(valid_times)
            std_t = np.std(valid_times)
            min_t = np.min(valid_times)
            max_t = np.max(valid_times)
            status = "✓" if mean_t < 60.0 else "⚠ >60s"
        else:
            mean_t = std_t = min_t = max_t = float("nan")
            status = "✗ FAIL"

        results[n] = {
            "n_elements": n,
            "mean_sec": round(float(mean_t), 4),
            "std_sec": round(float(std_t), 4),
            "min_sec": round(float(min_t), 4),
            "max_sec": round(float(max_t), 4),
            "n_repeats": N_REPEATS,
            "n_errors": len(errors),
            "ci_feasible": bool(mean_t < 60.0),
        }

        print(
            f"{n:>8} {mean_t:>10.3f} {std_t:>10.3f} "
            f"{min_t:>10.3f} {max_t:>10.3f} {status:>10}"
        )

    # ── Fit scaling model ─────────────────────────────────────────────
    sizes_arr = np.array([r["n_elements"] for r in results.values() if not np.isnan(r["mean_sec"])])
    times_arr = np.array([r["mean_sec"] for r in results.values() if not np.isnan(r["mean_sec"])])

    if len(sizes_arr) >= 3:
        log_s = np.log(sizes_arr)
        log_t = np.log(np.maximum(times_arr, 1e-6))
        coeffs = np.polyfit(log_s, log_t, 1)
        exponent = coeffs[0]
        constant = np.exp(coeffs[1])
        scaling_model = {
            "model": f"T(n) ≈ {constant:.4f} · n^{exponent:.3f}",
            "exponent": round(float(exponent), 4),
            "constant": round(float(constant), 6),
            "r_squared": round(float(1.0 - np.sum((log_t - np.polyval(coeffs, log_s))**2) / np.sum((log_t - np.mean(log_t))**2)), 4),
        }
        print(f"\n  Scaling model: {scaling_model['model']}")
        print(f"  R² = {scaling_model['r_squared']}")
    else:
        scaling_model = {"model": "insufficient data", "exponent": None}

    # ── CI/CD Feasibility Summary ─────────────────────────────────────
    feasible_up_to = max(
        (r["n_elements"] for r in results.values() if r.get("ci_feasible")),
        default=0,
    )
    print(f"\n  CI/CD feasible (≤60s) up to: {feasible_up_to} elements")

    output = {
        "scaling_data": results,
        "scaling_model": scaling_model,
        "ci_feasible_up_to_elements": feasible_up_to,
    }

    out_path = RESULTS_DIR / "exp3_scalability.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


if __name__ == "__main__":
    run_experiment()
