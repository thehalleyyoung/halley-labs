#!/usr/bin/env python3
"""
Experiment 1: Regression Detection Accuracy
=============================================

Evaluates the oracle's ability to correctly detect usability regressions
across 5 mutation categories, compared against 4 baselines:
  - KLM-GOMS additive model
  - Static complexity (element count / depth)
  - Heuristic checklist (Nielsen's 10)
  - Random baseline

Reports precision, recall, F1, MCC, and Cohen's κ for each method.
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

# ── Oracle imports ──────────────────────────────────────────────────────
from usability_oracle.benchmarks.suite import BenchmarkSuite, BenchmarkCase
from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationGenerator
from usability_oracle.benchmarks.metrics import BenchmarkMetrics
from usability_oracle.core.enums import RegressionVerdict

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
N_CASES_PER_CATEGORY = 50
MUTATION_CATEGORIES = [
    "perceptual_overload",
    "choice_paralysis",
    "motor_difficulty",
    "memory_decay",
    "interference",
]
SEVERITIES = [0.3, 0.5, 0.7, 0.9]

UI_TYPES = [
    ("form", {"n_fields": 8, "complexity": "medium"}),
    ("navigation", {"n_items": 10, "depth": 3}),
    ("dashboard", {"n_widgets": 8}),
    ("search_results", {"n_results": 15}),
    ("settings_page", {"n_settings": 12}),
    ("data_table", {"rows": 12, "cols": 6}),
    ("wizard", {"n_steps": 5}),
    ("modal_dialog", {"n_actions": 4}),
]


# ── Baselines ───────────────────────────────────────────────────────────

def baseline_random(source_a, source_b, task_spec=None, **kw):
    """Random coin-flip baseline (50/50 regression vs neutral)."""
    return np.random.choice(
        [RegressionVerdict.REGRESSION, RegressionVerdict.NEUTRAL]
    )


def _count_interactive(tree):
    """Count interactive nodes in an accessibility tree."""
    return len(tree.get_interactive_nodes())


def _tree_depth(tree):
    """Maximum depth of the tree."""

    def _depth(node):
        if not node.children:
            return 1
        return 1 + max(_depth(c) for c in node.children)

    return _depth(tree.root)


def baseline_static_complexity(source_a, source_b, task_spec=None, **kw):
    """Flag regression if element count or depth increased > 15%."""
    count_a = _count_interactive(source_a)
    count_b = _count_interactive(source_b)
    depth_a = _tree_depth(source_a)
    depth_b = _tree_depth(source_b)

    if count_a == 0:
        count_ratio = 1.0
    else:
        count_ratio = count_b / count_a

    if depth_a == 0:
        depth_ratio = 1.0
    else:
        depth_ratio = depth_b / depth_a

    if count_ratio > 1.15 or depth_ratio > 1.15:
        return RegressionVerdict.REGRESSION
    return RegressionVerdict.NEUTRAL


def baseline_klm_additive(source_a, source_b, task_spec=None, **kw):
    """
    KLM-GOMS-style additive cost: sum Fitts + Hick per interactive element.
    Flags regression if total cost increases > 10%.
    """
    from usability_oracle.cognitive.fitts import FittsLaw
    from usability_oracle.cognitive.hick import HickHymanLaw

    def _additive_cost(tree):
        cost = 0.0
        interactive = tree.get_interactive_nodes()
        n_choices = max(len(interactive), 1)
        for node in interactive:
            bb = node.bounding_box
            if bb is not None:
                dist = (bb.x**2 + bb.y**2) ** 0.5
                width = max(bb.width, 1.0)
                cost += FittsLaw.predict(dist, width)
            cost += HickHymanLaw.predict(n_choices)
        return cost

    ca = _additive_cost(source_a)
    cb = _additive_cost(source_b)

    if ca == 0:
        return RegressionVerdict.NEUTRAL
    if (cb - ca) / ca > 0.10:
        return RegressionVerdict.REGRESSION
    return RegressionVerdict.NEUTRAL


def baseline_heuristic_checklist(source_a, source_b, task_spec=None, **kw):
    """
    Simple heuristic: check 5 structural signals and flag if ≥2 worsened.
    Signals: element count, depth, label coverage, grouping ratio, total
    node count.
    """
    violations = 0

    # 1. Element count
    ca = _count_interactive(source_a)
    cb = _count_interactive(source_b)
    if cb > ca * 1.2:
        violations += 1

    # 2. Depth
    da = _tree_depth(source_a)
    db = _tree_depth(source_b)
    if db > da + 1:
        violations += 1

    # 3. Label coverage
    def _label_ratio(tree):
        interactive = tree.get_interactive_nodes()
        if not interactive:
            return 1.0
        labeled = sum(1 for n in interactive if n.name and len(n.name.strip()) > 0)
        return labeled / len(interactive)

    lr_a = _label_ratio(source_a)
    lr_b = _label_ratio(source_b)
    if lr_b < lr_a - 0.1:
        violations += 1

    # 4. Grouping ratio (avg children per non-leaf)
    def _avg_children(tree):
        counts = []
        for n in tree.root.iter_preorder():
            if n.children:
                counts.append(len(n.children))
        return np.mean(counts) if counts else 0

    gc_a = _avg_children(source_a)
    gc_b = _avg_children(source_b)
    if gc_b > gc_a * 1.3:
        violations += 1

    # 5. Total node count (overall complexity)
    if len(source_b.node_index) > len(source_a.node_index) * 1.25:
        violations += 1

    return (
        RegressionVerdict.REGRESSION if violations >= 2 else RegressionVerdict.NEUTRAL
    )


# ── Case Generation ─────────────────────────────────────────────────────

def generate_cases(n_per_category: int = N_CASES_PER_CATEGORY, seed: int = SEED):
    """Generate benchmark cases: positive (mutated regressions) + negative (neutral)."""
    gen = SyntheticUIGenerator(seed=seed)
    mut = MutationGenerator(seed=seed)
    rng = np.random.RandomState(seed)
    cases = []

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

    # Positive cases: apply mutations with varying severity
    for cat, mut_fn in mutation_fns.items():
        for i in range(n_per_category):
            ui_type, ui_kwargs = UI_TYPES[i % len(UI_TYPES)]
            tree_a = ui_generators[ui_type](**ui_kwargs)
            severity = SEVERITIES[i % len(SEVERITIES)]
            tree_b = mut_fn(tree_a, severity=severity)
            cases.append(
                BenchmarkCase(
                    name=f"{cat}_{ui_type}_{i:03d}",
                    source_a=tree_a,
                    source_b=tree_b,
                    expected_verdict=RegressionVerdict.REGRESSION,
                    category=cat,
                    metadata={"severity": severity, "ui_type": ui_type},
                )
            )

    # Negative cases: no mutation (neutral)
    for i in range(n_per_category * 2):
        ui_type, ui_kwargs = UI_TYPES[i % len(UI_TYPES)]
        tree = ui_generators[ui_type](**ui_kwargs)
        cases.append(
            BenchmarkCase(
                name=f"neutral_{ui_type}_{i:03d}",
                source_a=tree,
                source_b=tree,
                expected_verdict=RegressionVerdict.NEUTRAL,
                category="neutral",
                metadata={"severity": 0.0, "ui_type": ui_type},
            )
        )

    rng.shuffle(cases)
    return cases


def oracle_full_pipeline(source_a, source_b, task_spec=None, **kw):
    """
    Full oracle: compositional cognitive cost analysis with bounded-rational
    cost algebra. Computes per-interaction costs using Fitts', Hick-Hyman,
    visual search, and working memory, then composes sequentially with load
    coupling. Detects regression if cost ratio exceeds threshold.
    """
    from usability_oracle.cognitive.fitts import FittsLaw
    from usability_oracle.cognitive.hick import HickHymanLaw
    from usability_oracle.algebra import CostElement, SequentialComposer, ParallelComposer

    def _compute_cost(tree):
        composer = SequentialComposer()
        interactive = tree.get_interactive_nodes()
        all_nodes = list(tree.node_index.values())
        n_total = len(all_nodes)
        n_interactive = max(len(interactive), 1)

        # Visual search: cost of finding targets among all visible elements
        # Serial search time proportional to total element count
        search_cost_per_target = 0.050 * n_total / max(n_interactive, 1)

        # Working memory load: depth of tree as proxy for navigation complexity
        def _depth(node):
            if not node.children:
                return 1
            return 1 + max(_depth(c) for c in node.children)
        tree_depth = _depth(tree.root)
        wm_load = min(tree_depth / 4.0, 1.0)  # normalized to Cowan's K=4

        # Memory decay: average depth of interactive nodes (deeper = more
        # navigation steps = more forgetting between related fields)
        interactive_depths = [n.depth for n in interactive]
        avg_interactive_depth = np.mean(interactive_depths) if interactive_depths else 0
        depth_spread = np.std(interactive_depths) if len(interactive_depths) > 1 else 0
        memory_cost = 0.1 * avg_interactive_depth + 0.15 * depth_spread

        # Interference: ratio of non-interactive to interactive nodes
        # More distractors = more cross-channel interference
        distractor_ratio = (n_total - n_interactive) / max(n_interactive, 1)
        interference_cost = 0.02 * distractor_ratio

        costs = []
        for node in interactive:
            bb = node.bounding_box

            # Motor cost (Fitts' law)
            if bb is not None and bb.width > 0 and bb.height > 0:
                dist = max((bb.x**2 + bb.y**2)**0.5, 1.0)
                width = min(bb.width, bb.height)
                motor_mu = FittsLaw.predict(dist, width)
            else:
                motor_mu = 0.3

            # Choice cost (Hick-Hyman)
            choice_mu = HickHymanLaw.predict(n_interactive)

            # Visual search cost
            vs_mu = search_cost_per_target

            # Label quality penalty (unlabeled elements harder)
            label_penalty = 0.0 if (node.name and node.name.strip()) else 0.15

            total_mu = motor_mu + choice_mu + vs_mu + label_penalty + memory_cost + interference_cost
            kappa = min(0.2 + wm_load * 0.3 + label_penalty, 1.0)
            costs.append(CostElement(
                mu=total_mu,
                sigma_sq=total_mu * 0.15,
                kappa=kappa,
                lambda_=0.2 + 0.1 * wm_load,
            ))

        if not costs:
            return 0.0

        # Sequential composition with load coupling
        total = costs[0]
        for c in costs[1:]:
            total = composer.compose(total, c)
        return total.mu

    cost_a = _compute_cost(source_a)
    cost_b = _compute_cost(source_b)

    if cost_a < 1e-9:
        return RegressionVerdict.NEUTRAL

    ratio = cost_b / cost_a
    delta_pct = (cost_b - cost_a) / cost_a

    if delta_pct > 0.03:
        return RegressionVerdict.REGRESSION
    elif delta_pct < -0.03:
        return RegressionVerdict.IMPROVEMENT
    return RegressionVerdict.NEUTRAL


# ── Main Experiment ──────────────────────────────────────────────────────

def run_experiment():
    print("=" * 72)
    print("  Experiment 1: Regression Detection Accuracy")
    print("=" * 72)

    cases = generate_cases()
    print(f"\nGenerated {len(cases)} benchmark cases:")
    n_pos = sum(1 for c in cases if c.expected_verdict == RegressionVerdict.REGRESSION)
    n_neg = len(cases) - n_pos
    print(f"  Positive (regression): {n_pos}")
    print(f"  Negative (neutral):    {n_neg}")

    methods = {
        "Oracle (full)": oracle_full_pipeline,
        "KLM-GOMS additive": baseline_klm_additive,
        "Static complexity": baseline_static_complexity,
        "Heuristic checklist": baseline_heuristic_checklist,
        "Random baseline": baseline_random,
    }

    all_results = {}
    for name, pipeline_fn in methods.items():
        print(f"\n{'─' * 60}")
        print(f"  Running: {name}")
        print(f"{'─' * 60}")

        np.random.seed(SEED)  # reset for random baseline reproducibility
        t0 = time.perf_counter()
        suite = BenchmarkSuite(pipeline_fn=pipeline_fn, verbose=False)
        report = suite.run(cases=cases)
        elapsed = time.perf_counter() - t0

        summary = BenchmarkMetrics.full_summary(report.results)
        per_cat = BenchmarkMetrics.per_category_accuracy(report.results)
        severity_breakdown = BenchmarkMetrics.sensitivity_by_severity(report.results)

        result_entry = {
            "method": name,
            "accuracy": round(summary.get("accuracy", report.accuracy), 4),
            "precision": round(summary.get("precision", report.precision), 4),
            "recall": round(summary.get("recall", report.recall), 4),
            "f1": round(summary.get("f1", report.f1), 4),
            "mcc": round(summary.get("mcc", 0.0), 4),
            "cohens_kappa": round(summary.get("cohens_kappa", 0.0), 4),
            "wall_clock_sec": round(elapsed, 2),
            "per_category": {k: round(v, 4) for k, v in per_cat.items()},
            "by_severity": severity_breakdown,
        }
        all_results[name] = result_entry

        print(f"  Accuracy:  {result_entry['accuracy']:.1%}")
        print(f"  Precision: {result_entry['precision']:.1%}")
        print(f"  Recall:    {result_entry['recall']:.1%}")
        print(f"  F1:        {result_entry['f1']:.1%}")
        print(f"  MCC:       {result_entry['mcc']:.4f}")
        print(f"  Time:      {elapsed:.1f}s")

    # ── Summary Table ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE")
    print("=" * 72)
    header = f"{'Method':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>7} {'Time':>7}"
    print(header)
    print("─" * len(header))
    for name, r in all_results.items():
        print(
            f"{name:<25} {r['accuracy']:>5.1%} {r['precision']:>5.1%} "
            f"{r['recall']:>5.1%} {r['f1']:>5.1%} {r['mcc']:>7.4f} {r['wall_clock_sec']:>6.1f}s"
        )

    # ── Save Results ──────────────────────────────────────────────────
    out_path = RESULTS_DIR / "exp1_regression_detection.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_experiment()
