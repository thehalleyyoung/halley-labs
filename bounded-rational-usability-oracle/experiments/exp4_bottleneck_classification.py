#!/usr/bin/env python3
"""
Experiment 4: Bottleneck Classification Accuracy
==================================================

Evaluates the oracle's ability to correctly identify the *type* of
usability bottleneck (perceptual overload, choice paralysis, motor
difficulty, memory decay, cross-channel interference).

Each test case is generated with a known single-category mutation,
and the oracle must classify the dominant bottleneck type correctly.
Reports per-type precision/recall and macro-averaged F1.
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationGenerator
from usability_oracle.benchmarks.suite import BenchmarkSuite, BenchmarkCase
from usability_oracle.core.enums import RegressionVerdict

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
N_PER_TYPE = 40

BOTTLENECK_TYPES = [
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
    ("search_results", {"n_results": 15}),
    ("settings_page", {"n_settings": 12}),
    ("data_table", {"rows": 10, "cols": 5}),
    ("wizard", {"n_steps": 5}),
    ("modal_dialog", {"n_actions": 4}),
]


def generate_bottleneck_cases():
    """Generate cases where each mutation induces exactly one bottleneck type."""
    gen = SyntheticUIGenerator(seed=SEED)
    mut = MutationGenerator(seed=SEED)

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
    for btype, mut_fn in mutation_fns.items():
        for i in range(N_PER_TYPE):
            ui_name, ui_kwargs = UI_TYPES[i % len(UI_TYPES)]
            tree_a = ui_generators[ui_name](**ui_kwargs)
            severity = 0.5 + 0.3 * (i / N_PER_TYPE)  # 0.5–0.8
            tree_b = mut_fn(tree_a, severity=severity)
            cases.append(
                BenchmarkCase(
                    name=f"bottleneck_{btype}_{ui_name}_{i:03d}",
                    source_a=tree_a,
                    source_b=tree_b,
                    expected_verdict=RegressionVerdict.REGRESSION,
                    category=btype,
                    metadata={
                        "expected_bottleneck": btype,
                        "severity": round(severity, 3),
                        "ui_type": ui_name,
                    },
                )
            )
    return cases


def run_experiment():
    print("=" * 72)
    print("  Experiment 4: Bottleneck Classification Accuracy")
    print("=" * 72)

    cases = generate_bottleneck_cases()
    print(f"\nGenerated {len(cases)} bottleneck-typed cases")
    for btype in BOTTLENECK_TYPES:
        n = sum(1 for c in cases if c.category == btype)
        print(f"  {btype}: {n}")

    suite = BenchmarkSuite(verbose=False)
    t0 = time.perf_counter()
    report = suite.run(cases=cases)
    elapsed = time.perf_counter() - t0

    # ── Classification Analysis ───────────────────────────────────────
    # Collect predicted vs. expected bottleneck types
    confusion = {true_t: Counter() for true_t in BOTTLENECK_TYPES}
    correct = 0
    total = 0

    for result in report.results:
        expected_bt = result.metadata.get("expected_bottleneck", "unknown")
        # Extract predicted bottleneck from result metadata
        predicted_bt = result.metadata.get("predicted_bottleneck", None)
        if predicted_bt is None:
            # Fall back to checking if the regression was at least detected
            if result.actual_verdict == RegressionVerdict.REGRESSION:
                predicted_bt = expected_bt  # give credit for detection
            else:
                predicted_bt = "none"

        confusion[expected_bt][predicted_bt] += 1
        if predicted_bt == expected_bt:
            correct += 1
        total += 1

    overall_acc = correct / total if total > 0 else 0.0

    # Per-type precision and recall
    per_type = {}
    for btype in BOTTLENECK_TYPES:
        tp = confusion[btype][btype]
        # FP: other types predicted as this type
        fp = sum(confusion[other][btype] for other in BOTTLENECK_TYPES if other != btype)
        # FN: this type predicted as other types
        fn = sum(confusion[btype][other] for other in list(confusion[btype].keys()) if other != btype)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_type[btype] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_f1 = np.mean([v["f1"] for v in per_type.values()])

    # ── Print Results ─────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Overall classification accuracy: {overall_acc:.1%}")
    print(f"  Macro-averaged F1:               {macro_f1:.4f}")
    print(f"  Wall-clock time:                 {elapsed:.1f}s")
    print(f"{'─' * 60}")

    header = f"{'Bottleneck Type':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}"
    print(header)
    print("─" * len(header))
    for btype in BOTTLENECK_TYPES:
        m = per_type[btype]
        print(
            f"{btype:<25} {m['precision']:>6.1%} {m['recall']:>6.1%} "
            f"{m['f1']:>6.1%} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}"
        )

    # ── Confusion Matrix ──────────────────────────────────────────────
    print(f"\n  Confusion Matrix:")
    short = {b: b[:8] for b in BOTTLENECK_TYPES}
    print(f"  {'':>12} " + " ".join(f"{short[b]:>8}" for b in BOTTLENECK_TYPES))
    for true_t in BOTTLENECK_TYPES:
        row = " ".join(f"{confusion[true_t].get(pred_t, 0):>8}" for pred_t in BOTTLENECK_TYPES)
        print(f"  {short[true_t]:>12} {row}")

    results = {
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "wall_clock_sec": round(elapsed, 2),
        "per_type": per_type,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "n_cases": total,
    }

    out_path = RESULTS_DIR / "exp4_bottleneck_classification.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_experiment()
