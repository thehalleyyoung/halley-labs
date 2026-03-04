#!/usr/bin/env python3
"""
Experiment 5: Ablation Study — Component Contributions
========================================================

Systematically disables each major component of the oracle cost
function to quantify its marginal contribution to regression detection.

Ablation variants:
  1. Full system (all components)
  2. No Fitts' law (motor cost set to constant)
  3. No Hick-Hyman (choice cost set to constant)
  4. No visual search model
  5. No working memory model
  6. No interference model
  7. Fitts only (all other components disabled)
  8. Hick only (all other components disabled)
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationGenerator
from usability_oracle.accessibility.models import AccessibilityTree, AccessibilityNode
from usability_oracle.cognitive.fitts import FittsLaw
from usability_oracle.cognitive.hick import HickHymanLaw
from usability_oracle.algebra import CostElement, SequentialComposer

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
N_PER_CATEGORY = 30

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


def compute_oracle_cost(tree: AccessibilityTree, disable: set[str] | None = None) -> float:
    """Compute oracle cost with optional component ablation."""
    disable = disable or set()
    interactive = tree.get_interactive_nodes()
    all_nodes = list(tree.node_index.values())
    n_all = max(len(all_nodes), 1)
    n_interactive = max(len(interactive), 1)
    composer = SequentialComposer()

    elements = []
    for node in interactive:
        mu = 0.0

        # Fitts' law (motor)
        if "fitts" not in disable:
            if node.bounding_box:
                w = max(node.bounding_box.width, 1)
                d = math.sqrt(node.bounding_box.x**2 + node.bounding_box.y**2) + 1
                mu += FittsLaw.predict(distance=d, width=w)
            else:
                mu += 0.3

        # Visual search
        if "visual_search" not in disable:
            distractors = max(n_all - 1, 0)
            mu += 0.05 * math.log2(distractors + 2)

        # Working memory
        if "working_memory" not in disable:
            depth = node.depth if hasattr(node, "depth") else 1
            mu += 0.1 * max(depth - 2, 0)

        # Interference
        if "interference" not in disable:
            distractor_ratio = max(n_all - n_interactive, 0) / n_all
            mu += 0.15 * distractor_ratio

        # Label quality
        if "label" not in disable:
            name = getattr(node, "name", "") or ""
            if len(name) < 2:
                mu += 0.2

        elements.append(CostElement(mu=mu, sigma_sq=0.01, kappa=0.3, lambda_=0.1))

    # Hick-Hyman (choice)
    if "hick" not in disable:
        hick_cost = HickHymanLaw.predict(n_interactive)
        elements.append(CostElement(mu=hick_cost, sigma_sq=0.02, kappa=0.5, lambda_=0.2))

    if not elements:
        return 0.0

    total = elements[0]
    for e in elements[1:]:
        total = composer.compose(total, e)
    return total.mu


REGRESSION_THRESHOLD = 0.03

ABLATION_VARIANTS = {
    "Full system": set(),
    "No Fitts' law": {"fitts"},
    "No Hick-Hyman": {"hick"},
    "No visual search": {"visual_search"},
    "No working memory": {"working_memory"},
    "No interference": {"interference"},
    "Fitts only": {"hick", "visual_search", "working_memory", "interference", "label"},
    "Hick only": {"fitts", "visual_search", "working_memory", "interference", "label"},
}


def generate_ablation_cases():
    """Generate positive (regression) and negative (no-change) cases."""
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
        for i in range(N_PER_CATEGORY):
            ui_name, ui_kwargs = UI_TYPES[i % len(UI_TYPES)]
            tree_a = ui_generators[ui_name](**ui_kwargs)
            severity = 0.4 + 0.4 * (i / N_PER_CATEGORY)
            tree_b = mut_fn(tree_a, severity=severity)
            cases.append({
                "tree_a": tree_a,
                "tree_b": tree_b,
                "is_regression": True,
                "category": cat,
            })

    # Negative cases (same tree twice)
    for i in range(N_PER_CATEGORY * 2):
        ui_name, ui_kwargs = UI_TYPES[i % len(UI_TYPES)]
        tree = ui_generators[ui_name](**ui_kwargs)
        cases.append({
            "tree_a": tree,
            "tree_b": tree,
            "is_regression": False,
            "category": "neutral",
        })

    rng.shuffle(cases)
    return cases


def evaluate_variant(cases, disable: set[str]):
    """Run oracle with given ablation and compute metrics."""
    tp = fp = tn = fn = 0
    for case in cases:
        cost_a = compute_oracle_cost(case["tree_a"], disable=disable)
        cost_b = compute_oracle_cost(case["tree_b"], disable=disable)
        base = max(cost_a, 0.001)
        delta_pct = (cost_b - cost_a) / base
        predicted_reg = delta_pct > REGRESSION_THRESHOLD
        actual_reg = case["is_regression"]

        if predicted_reg and actual_reg:
            tp += 1
        elif predicted_reg and not actual_reg:
            fp += 1
        elif not predicted_reg and actual_reg:
            fn += 1
        else:
            tn += 1

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    # MCC
    num = tp * tn - fp * fn
    denom = math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1))
    mcc = num / denom

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_experiment():
    print("=" * 72)
    print("  Experiment 5: Ablation Study — Component Contributions")
    print("=" * 72)

    cases = generate_ablation_cases()
    n_pos = sum(1 for c in cases if c["is_regression"])
    n_neg = len(cases) - n_pos
    print(f"\nGenerated {len(cases)} cases ({n_pos} positive, {n_neg} negative)")

    all_results = {}
    for variant_name, disable in ABLATION_VARIANTS.items():
        print(f"\n{'─' * 60}")
        print(f"  Variant: {variant_name}")
        print(f"{'─' * 60}")

        t0 = time.perf_counter()
        metrics = evaluate_variant(cases, disable)
        elapsed = time.perf_counter() - t0

        metrics["wall_clock_sec"] = round(elapsed, 2)
        metrics["disabled"] = sorted(disable)
        all_results[variant_name] = metrics

        print(f"  Accuracy:  {metrics['accuracy']:.1%}")
        print(f"  Precision: {metrics['precision']:.1%}")
        print(f"  Recall:    {metrics['recall']:.1%}")
        print(f"  F1:        {metrics['f1']:.1%}")
        print(f"  MCC:       {metrics['mcc']:.4f}")
        print(f"  Time:      {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 72)
    print("  ABLATION SUMMARY")
    print("=" * 72)
    header = f"{'Variant':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>7} {'ΔF1':>7}"
    print(header)
    print("─" * len(header))

    full_f1 = all_results["Full system"]["f1"]
    for name, r in all_results.items():
        delta = r["f1"] - full_f1
        sign = "+" if delta >= 0 else ""
        print(
            f"{name:<30} {r['accuracy']:>5.1%} {r['precision']:>5.1%} "
            f"{r['recall']:>5.1%} {r['f1']:>5.1%} {r['mcc']:>6.4f} {sign}{delta:>6.1%}"
        )

    # Component importance
    print(f"\n  Component importance (F1 drop from full system):")
    importance = {}
    for name, r in all_results.items():
        if name != "Full system":
            drop = full_f1 - r["f1"]
            importance[name] = drop
    for name, drop in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * max(int(drop * 200), 0)
        sign = "−" if drop > 0 else "+"
        print(f"    {name:<30} {sign}{abs(drop):.1%} {bar}")

    out_path = RESULTS_DIR / "exp5_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    run_experiment()
