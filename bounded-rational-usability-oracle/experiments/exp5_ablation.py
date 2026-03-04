#!/usr/bin/env python3
"""
Experiment 5: Ablation Study — Component Contributions
========================================================

Systematically disables each major component of the oracle to quantify
its marginal contribution to regression detection accuracy.

Ablation variants:
  1. Full system (all components)
  2. No bisimulation (skip state abstraction → raw MDP)
  3. No cost algebra (additive costs only, no ⊕/⊗/Δ operators)
  4. No bottleneck taxonomy (detection only, no classification)
  5. No Monte Carlo sampling (single-trajectory deterministic)
  6. No working memory model
  7. No visual search model
  8. Minimal: Fitts + Hick additive only
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
N_CASES_PER_CATEGORY = 30

MUTATION_CATEGORIES = [
    "perceptual_overload",
    "choice_paralysis",
    "motor_difficulty",
    "memory_decay",
    "interference",
]

UI_TYPES = [
    ("form", {"n_fields": 8, "complexity": "medium"}),
    ("navigation", {"n_items": 10, "depth": 3}),
    ("dashboard", {"n_widgets": 8}),
    ("search_results", {"n_results": 12}),
    ("settings_page", {"n_settings": 10}),
    ("data_table", {"rows": 10, "cols": 5}),
    ("wizard", {"n_steps": 4}),
    ("modal_dialog", {"n_actions": 3}),
]


def generate_ablation_cases():
    gen = SyntheticUIGenerator(seed=SEED)
    mut = MutationGenerator(seed=SEED)
    rng = np.random.RandomState(SEED)

    mutation_fns = {
        "perceptual_overload": mut.apply_perceptual_overload,
        "choice_paralysis": mut.apply_choice_paralysis,
        "motor_difficulty": mut.apply_motor_difficulty,
        "memory_decay": mut.apply_memory_decay,
        "interference": mut.apply_interference,
    }
    ui_generators = {
        "form": gen.generate_form,
        "navigation": gen.generate_navigation,
        "dashboard": gen.generate_dashboard,
        "search_results": gen.generate_search_results,
        "settings_page": gen.generate_settings_page,
        "data_table": gen.generate_data_table,
        "wizard": gen.generate_wizard,
        "modal_dialog": gen.generate_modal_dialog,
    }

    cases = []
    # Positive cases
    for cat, mut_fn in mutation_fns.items():
        for i in range(N_CASES_PER_CATEGORY):
            ui_name, ui_kwargs = UI_TYPES[i % len(UI_TYPES)]
            tree_a = ui_generators[ui_name](**ui_kwargs)
            severity = 0.4 + 0.4 * (i / N_CASES_PER_CATEGORY)
            tree_b = mut_fn(tree_a, severity=severity)
            cases.append(
                BenchmarkCase(
                    name=f"abl_{cat}_{i:03d}",
                    source_a=tree_a,
                    source_b=tree_b,
                    expected_verdict=RegressionVerdict.REGRESSION,
                    category=cat,
                    metadata={"severity": round(severity, 3)},
                )
            )

    # Negative cases
    for i in range(N_CASES_PER_CATEGORY * 2):
        ui_name, ui_kwargs = UI_TYPES[i % len(UI_TYPES)]
        tree = ui_generators[ui_name](**ui_kwargs)
        cases.append(
            BenchmarkCase(
                name=f"abl_neutral_{i:03d}",
                source_a=tree,
                source_b=tree,
                expected_verdict=RegressionVerdict.NEUTRAL,
                category="neutral",
            )
        )

    rng.shuffle(cases)
    return cases


# ── Ablation configurations ─────────────────────────────────────────────

ABLATION_CONFIGS = {
    "Full system": {},
    "No bisimulation": {"bisimulation": {"enabled": False}},
    "No cost algebra (additive only)": {"algebra": {"mode": "additive"}},
    "No bottleneck taxonomy": {"bottleneck": {"enabled": False}},
    "No Monte Carlo (deterministic)": {"montecarlo": {"n_trajectories": 1}},
    "No working memory": {"cognitive": {"working_memory": False}},
    "No visual search": {"cognitive": {"visual_search": False}},
    "Minimal (Fitts+Hick only)": {
        "bisimulation": {"enabled": False},
        "algebra": {"mode": "additive"},
        "bottleneck": {"enabled": False},
        "cognitive": {"working_memory": False, "visual_search": False},
        "montecarlo": {"n_trajectories": 1},
    },
}


def run_experiment():
    print("=" * 72)
    print("  Experiment 5: Ablation Study — Component Contributions")
    print("=" * 72)

    cases = generate_ablation_cases()
    n_pos = sum(1 for c in cases if c.expected_verdict == RegressionVerdict.REGRESSION)
    n_neg = len(cases) - n_pos
    print(f"\nGenerated {len(cases)} cases ({n_pos} positive, {n_neg} negative)")

    all_results = {}
    for variant_name, config in ABLATION_CONFIGS.items():
        print(f"\n{'─' * 60}")
        print(f"  Variant: {variant_name}")
        print(f"{'─' * 60}")

        t0 = time.perf_counter()
        suite = BenchmarkSuite(verbose=False)
        report = suite.run(cases=cases, config=config)
        elapsed = time.perf_counter() - t0

        summary = BenchmarkMetrics.full_summary(report.results)
        per_cat = BenchmarkMetrics.per_category_accuracy(report.results)

        entry = {
            "variant": variant_name,
            "accuracy": round(summary.get("accuracy", report.accuracy), 4),
            "precision": round(summary.get("precision", report.precision), 4),
            "recall": round(summary.get("recall", report.recall), 4),
            "f1": round(summary.get("f1", report.f1), 4),
            "mcc": round(summary.get("mcc", 0.0), 4),
            "wall_clock_sec": round(elapsed, 2),
            "per_category": {k: round(v, 4) for k, v in per_cat.items()},
            "config": config,
        }
        all_results[variant_name] = entry

        print(f"  Accuracy:  {entry['accuracy']:.1%}")
        print(f"  Precision: {entry['precision']:.1%}")
        print(f"  Recall:    {entry['recall']:.1%}")
        print(f"  F1:        {entry['f1']:.1%}")
        print(f"  Time:      {elapsed:.1f}s")

    # ── Summary Table ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ABLATION SUMMARY")
    print("=" * 72)
    header = f"{'Variant':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'ΔF1':>7}"
    print(header)
    print("─" * len(header))

    full_f1 = all_results["Full system"]["f1"]
    for name, r in all_results.items():
        delta = r["f1"] - full_f1
        sign = "+" if delta >= 0 else ""
        print(
            f"{name:<35} {r['accuracy']:>5.1%} {r['precision']:>5.1%} "
            f"{r['recall']:>5.1%} {r['f1']:>5.1%} {sign}{delta:>6.1%}"
        )

    # ── Component Importance (F1 drop) ────────────────────────────────
    print(f"\n  Component importance (F1 drop from full system):")
    importance = {}
    for name, r in all_results.items():
        if name != "Full system":
            drop = full_f1 - r["f1"]
            importance[name] = drop
    for name, drop in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(drop * 100)
        print(f"    {name:<35} −{drop:.1%} {bar}")

    out_path = RESULTS_DIR / "exp5_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    run_experiment()
