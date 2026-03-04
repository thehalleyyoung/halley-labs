#!/usr/bin/env python3
"""
Experiment 4: Bottleneck Classification Accuracy
==================================================

Evaluates the oracle's ability to correctly identify the *type* of
usability bottleneck (perceptual overload, choice paralysis, motor
difficulty, memory decay, cross-channel interference).

Each test case is generated with a known single-category mutation,
and the oracle must classify the dominant bottleneck type correctly
based on which cost component increased the most.

Reports per-type precision/recall and macro-averaged F1.
"""

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationGenerator
from usability_oracle.accessibility.models import AccessibilityTree, AccessibilityNode
from usability_oracle.cognitive.fitts import FittsLaw
from usability_oracle.cognitive.hick import HickHymanLaw

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


def _compute_cost_components(tree: AccessibilityTree) -> dict:
    """Compute per-component cost breakdown for a tree."""
    interactive = tree.get_interactive_nodes()
    if not interactive:
        return {"fitts": 0.0, "hick": 0.0, "visual_search": 0.0,
                "working_memory": 0.0, "interference": 0.0}

    n_choices = max(len(interactive), 1)
    total_fitts = 0.0
    total_visual = 0.0
    total_wm = 0.0
    total_interference = 0.0

    all_nodes = list(tree.node_index.values())
    n_all = max(len(all_nodes), 1)

    for node in interactive:
        # Fitts (motor)
        if node.bounding_box:
            w = max(node.bounding_box.width, 1)
            d = math.sqrt(node.bounding_box.x**2 + node.bounding_box.y**2) + 1
            total_fitts += FittsLaw.predict(distance=d, width=w)
        else:
            total_fitts += 0.3

        # Visual search (distractors)
        depth = node.depth if hasattr(node, "depth") else 1
        distractors = max(n_all - 1, 0)
        total_visual += 0.05 * math.log2(distractors + 2)

        # Working memory (depth)
        total_wm += 0.1 * max(depth - 2, 0)

        # Interference (distractor ratio)
        distractor_ratio = max(n_all - len(interactive), 0) / n_all
        total_interference += 0.15 * distractor_ratio

    hick = HickHymanLaw.predict(n_choices)

    return {
        "fitts": total_fitts,
        "hick": hick,
        "visual_search": total_visual,
        "working_memory": total_wm,
        "interference": total_interference,
    }


# Mapping from cost component to bottleneck type
COMPONENT_TO_TYPE = {
    "visual_search": "perceptual_overload",
    "hick": "choice_paralysis",
    "fitts": "motor_difficulty",
    "working_memory": "memory_decay",
    "interference": "interference",
}


def classify_bottleneck(tree_a: AccessibilityTree, tree_b: AccessibilityTree) -> str:
    """Classify bottleneck type based on which cost component increased most."""
    costs_a = _compute_cost_components(tree_a)
    costs_b = _compute_cost_components(tree_b)

    deltas = {}
    for comp in costs_a:
        base = max(costs_a[comp], 0.001)
        deltas[comp] = (costs_b[comp] - costs_a[comp]) / base

    max_comp = max(deltas, key=lambda k: deltas[k])
    return COMPONENT_TO_TYPE.get(max_comp, "unknown")


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
            severity = 0.5 + 0.3 * (i / N_PER_TYPE)
            tree_b = mut_fn(tree_a, severity=severity)
            cases.append({
                "name": f"bottleneck_{btype}_{ui_name}_{i:03d}",
                "tree_a": tree_a,
                "tree_b": tree_b,
                "expected_type": btype,
                "severity": round(severity, 3),
                "ui_type": ui_name,
            })
    return cases


def run_experiment():
    print("=" * 72)
    print("  Experiment 4: Bottleneck Classification Accuracy")
    print("=" * 72)

    cases = generate_bottleneck_cases()
    print(f"\nGenerated {len(cases)} bottleneck-typed cases")
    for btype in BOTTLENECK_TYPES:
        n = sum(1 for c in cases if c["expected_type"] == btype)
        print(f"  {btype}: {n}")

    t0 = time.perf_counter()

    confusion = {true_t: Counter() for true_t in BOTTLENECK_TYPES}
    correct = 0
    total = 0

    for case in cases:
        predicted = classify_bottleneck(case["tree_a"], case["tree_b"])
        expected = case["expected_type"]
        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1
        total += 1

    elapsed = time.perf_counter() - t0
    overall_acc = correct / total if total > 0 else 0.0

    per_type = {}
    for btype in BOTTLENECK_TYPES:
        tp = confusion[btype][btype]
        fp = sum(confusion[other][btype] for other in BOTTLENECK_TYPES if other != btype)
        fn = sum(confusion[btype][other] for other in confusion[btype] if other != btype)

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
