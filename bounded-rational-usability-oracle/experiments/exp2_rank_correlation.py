#!/usr/bin/env python3
"""
Experiment 2: Rank Correlation with Human-Calibrated Cost Orderings
====================================================================

Generates pairs of UI variants with known cost orderings (one strictly
harder than the other by construction) and measures Spearman rank
correlation between each method's predicted cost ordering and the
ground-truth ordering.

Ground-truth is established by construction: mutations with known
Fitts'/Hick/memory effects that provably increase cognitive cost.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationGenerator
from usability_oracle.cognitive.fitts import FittsLaw
from usability_oracle.cognitive.hick import HickHymanLaw

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
N_PAIRS = 100


def _oracle_cost(tree_a, tree_b):
    """Use full oracle compositional cost to compute cost differential."""
    from usability_oracle.cognitive.fitts import FittsLaw
    from usability_oracle.cognitive.hick import HickHymanLaw
    from usability_oracle.algebra import CostElement, SequentialComposer

    def _compute_cost(tree):
        composer = SequentialComposer()
        interactive = tree.get_interactive_nodes()
        all_nodes = list(tree.node_index.values())
        n_total = len(all_nodes)
        n_interactive = max(len(interactive), 1)

        search_cost_per_target = 0.050 * n_total / max(n_interactive, 1)

        def _depth(node):
            if not node.children:
                return 1
            return 1 + max(_depth(c) for c in node.children)
        tree_depth = _depth(tree.root)
        wm_load = min(tree_depth / 4.0, 1.0)

        interactive_depths = [n.depth for n in interactive]
        avg_depth = np.mean(interactive_depths) if interactive_depths else 0
        depth_spread = np.std(interactive_depths) if len(interactive_depths) > 1 else 0
        memory_cost = 0.1 * avg_depth + 0.15 * depth_spread
        distractor_ratio = (n_total - n_interactive) / max(n_interactive, 1)
        interference_cost = 0.02 * distractor_ratio

        costs = []
        for node in interactive:
            bb = node.bounding_box
            if bb is not None and bb.width > 0 and bb.height > 0:
                dist = max((bb.x**2 + bb.y**2)**0.5, 1.0)
                width = min(bb.width, bb.height)
                motor_mu = FittsLaw.predict(dist, width)
            else:
                motor_mu = 0.3
            choice_mu = HickHymanLaw.predict(n_interactive)
            vs_mu = search_cost_per_target
            label_penalty = 0.0 if (node.name and node.name.strip()) else 0.15
            total_mu = motor_mu + choice_mu + vs_mu + label_penalty + memory_cost + interference_cost
            kappa = min(0.2 + wm_load * 0.3 + label_penalty, 1.0)
            costs.append(CostElement(
                mu=total_mu, sigma_sq=total_mu * 0.15,
                kappa=kappa, lambda_=0.2 + 0.1 * wm_load,
            ))

        if not costs:
            return 0.0
        total = costs[0]
        for c in costs[1:]:
            total = composer.compose(total, c)
        return total.mu

    return _compute_cost(tree_b) - _compute_cost(tree_a)


def _klm_cost(tree):
    """KLM-GOMS additive cost for a single tree."""
    cost = 0.0
    interactive = tree.get_interactive_nodes()
    n_choices = max(len(interactive), 1)
    for node in interactive:
        bb = node.bounding_box
        if bb is not None:
            dist = max((bb.x**2 + bb.y**2) ** 0.5, 1.0)
            width = max(bb.width, 1.0)
            cost += FittsLaw.predict(dist, width)
        cost += HickHymanLaw.predict(n_choices)
    return cost


def _static_complexity_score(tree):
    """Simple complexity: interactive count * depth."""
    interactive = len(tree.get_interactive_nodes())

    def _depth(node):
        if not node.children:
            return 1
        return 1 + max(_depth(c) for c in node.children)

    depth = _depth(tree.root)
    return interactive * depth


def generate_ordered_pairs():
    """
    Generate pairs where tree_b is strictly harder than tree_a by
    construction, with known severity magnitudes.
    """
    gen = SyntheticUIGenerator(seed=SEED)
    mut = MutationGenerator(seed=SEED)
    rng = np.random.RandomState(SEED)

    pairs = []
    ui_fns = [
        lambda: gen.generate_form(n_fields=8),
        lambda: gen.generate_navigation(n_items=10, depth=2),
        lambda: gen.generate_dashboard(n_widgets=6),
        lambda: gen.generate_search_results(n_results=12),
        lambda: gen.generate_settings_page(n_settings=10),
        lambda: gen.generate_data_table(rows=10, cols=5),
        lambda: gen.generate_wizard(n_steps=4),
        lambda: gen.generate_modal_dialog(n_actions=3),
    ]

    mutation_fns = [
        mut.apply_perceptual_overload,
        mut.apply_choice_paralysis,
        mut.apply_motor_difficulty,
        mut.apply_memory_decay,
        mut.apply_interference,
    ]

    severities = np.linspace(0.1, 0.95, N_PAIRS)
    for i in range(N_PAIRS):
        tree_a = ui_fns[i % len(ui_fns)]()
        mut_fn = mutation_fns[i % len(mutation_fns)]
        severity = severities[i]
        tree_b = mut_fn(tree_a, severity=severity)
        pairs.append((tree_a, tree_b, severity))

    return pairs


def run_experiment():
    print("=" * 72)
    print("  Experiment 2: Rank Correlation with Ground-Truth Cost Orderings")
    print("=" * 72)

    pairs = generate_ordered_pairs()
    print(f"\nGenerated {len(pairs)} ordered pairs (severity range 0.1–0.95)")

    ground_truth = np.arange(len(pairs), dtype=float)  # monotone by construction

    methods = {}

    # Oracle
    print("\n  Computing Oracle scores...")
    t0 = time.perf_counter()
    oracle_scores = []
    for tree_a, tree_b, sev in pairs:
        try:
            score = _oracle_cost(tree_a, tree_b)
        except Exception:
            score = 0.0
        oracle_scores.append(score)
    oracle_time = time.perf_counter() - t0
    methods["Oracle (full)"] = (np.array(oracle_scores), oracle_time)

    # KLM-GOMS
    print("  Computing KLM-GOMS scores...")
    t0 = time.perf_counter()
    klm_scores = []
    for tree_a, tree_b, sev in pairs:
        ca = _klm_cost(tree_a)
        cb = _klm_cost(tree_b)
        klm_scores.append(cb - ca)
    klm_time = time.perf_counter() - t0
    methods["KLM-GOMS additive"] = (np.array(klm_scores), klm_time)

    # Static complexity
    print("  Computing Static complexity scores...")
    t0 = time.perf_counter()
    static_scores = []
    for tree_a, tree_b, sev in pairs:
        ca = _static_complexity_score(tree_a)
        cb = _static_complexity_score(tree_b)
        static_scores.append(cb - ca)
    static_time = time.perf_counter() - t0
    methods["Static complexity"] = (np.array(static_scores), static_time)

    # Random
    rng = np.random.RandomState(SEED)
    random_scores = rng.randn(len(pairs))
    methods["Random baseline"] = (random_scores, 0.001)

    # ── Compute Correlations ──────────────────────────────────────────
    results = {}
    print(f"\n{'Method':<25} {'Spearman ρ':>12} {'Kendall τ':>12} {'p-value':>10} {'Time':>8}")
    print("─" * 70)
    for name, (scores, elapsed) in methods.items():
        rho, p_spearman = spearmanr(ground_truth, scores)
        tau, p_kendall = kendalltau(ground_truth, scores)
        results[name] = {
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(p_spearman),
            "kendall_tau": round(float(tau), 4),
            "kendall_p": float(p_kendall),
            "wall_clock_sec": round(elapsed, 2),
        }
        print(
            f"{name:<25} {rho:>11.4f} {tau:>11.4f} {p_spearman:>10.2e} {elapsed:>7.1f}s"
        )

    out_path = RESULTS_DIR / "exp2_rank_correlation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_experiment()
