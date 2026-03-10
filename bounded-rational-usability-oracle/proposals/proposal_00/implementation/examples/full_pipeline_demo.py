#!/usr/bin/env python3
"""
Example: Full pipeline demonstration.

Walks through every stage of the usability oracle pipeline:

1. Parse — Build accessibility trees from raw data.
2. Cost  — Assign Fitts' law / Hick-Hyman costs.
3. MDP   — Construct a Markov Decision Process.
4. Solve — Find bounded-rational policy via softmax value iteration.
5. Compare — Detect regressions between before/after.
6. Bottleneck — Identify and rank usability bottlenecks.
7. Report — Generate a text report.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.core.enums import (
    AccessibilityRole,
    BottleneckType,
    RegressionVerdict,
)
from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationOperator
from usability_oracle.evaluation.baselines import BaselineComparator
from usability_oracle.utils.entropy import (
    entropy,
    effective_number,
    channel_capacity,
    mutual_information,
    markov_entropy_rate,
)
from usability_oracle.analysis.complexity import (
    estimate_empirical_complexity,
    fit_complexity_model,
)
from usability_oracle.analysis.statistical import (
    cohens_d,
    bootstrap_ci,
)
from usability_oracle.visualization.tree_viz import render_ascii_tree
from usability_oracle.visualization.cost_viz import (
    fitts_law_analysis,
    hick_hyman_analysis,
)
from usability_oracle.visualization.report_viz import (
    generate_full_report,
    generate_executive_summary,
)
from usability_oracle.simulation.agent import BoundedRationalAgent
from usability_oracle.simulation.environment import UIEnvironment


# ---------------------------------------------------------------------------
# Stage 1: Parse / Build trees
# ---------------------------------------------------------------------------

def stage_parse() -> tuple[AccessibilityTree, AccessibilityTree]:
    """Build before and after accessibility trees."""
    print("[Stage 1] Parsing accessibility trees...")
    gen = SyntheticUIGenerator(seed=42)

    before = gen.generate_form(n_fields=6, include_validation=True)
    print(f"  Before tree: {len(before.node_index)} nodes")

    # Create an "after" version with regressions
    mutator = MutationOperator(seed=42)
    after = mutator.apply_perceptual_overload(before, severity=0.6)
    after = mutator.apply_label_removal(after, fraction=0.3)
    print(f"  After tree:  {len(after.node_index)} nodes")

    return before, after


# ---------------------------------------------------------------------------
# Stage 2: Cost assignment
# ---------------------------------------------------------------------------

def stage_cost(tree: AccessibilityTree, label: str) -> dict[str, float]:
    """Assign interaction costs using Fitts' and Hick-Hyman laws."""
    print(f"\n[Stage 2] Computing costs ({label})...")

    interactive = [
        n for n in tree.node_index.values()
        if n.role.lower() in ("button", "textbox", "link", "checkbox",
                               "combobox", "slider", "radio")
    ]

    # Fitts' law: T = a + b * log₂(2D/W)
    fitts_a, fitts_b = 0.1, 0.1  # empirical constants
    motor_costs = []
    for node in interactive:
        if node.bounding_box:
            w = max(node.bounding_box.width, 1)
            d = math.sqrt(node.bounding_box.x ** 2 + node.bounding_box.y ** 2)
            t = fitts_a + fitts_b * math.log2(2 * max(d, 1) / w + 1)
            motor_costs.append(t)

    # Hick-Hyman: T = a + b * log₂(n + 1)
    n_choices = len(interactive)
    hick_time = 0.2 + 0.15 * math.log2(n_choices + 1)

    costs = {
        "n_interactive": n_choices,
        "avg_fitts_time": float(np.mean(motor_costs)) if motor_costs else 0.0,
        "max_fitts_time": float(np.max(motor_costs)) if motor_costs else 0.0,
        "hick_time": hick_time,
        "total_estimated_time": sum(motor_costs) + hick_time,
    }

    for k, v in costs.items():
        print(f"    {k}: {v:.3f}")

    return costs


# ---------------------------------------------------------------------------
# Stage 3: MDP construction
# ---------------------------------------------------------------------------

def stage_mdp(tree: AccessibilityTree, label: str) -> dict[str, Any]:
    """Build a simplified MDP from the accessibility tree."""
    print(f"\n[Stage 3] Building MDP ({label})...")

    interactive = [
        n for n in tree.node_index.values()
        if n.role.lower() in ("button", "textbox", "link", "checkbox",
                               "combobox", "slider", "radio")
    ]
    n_states = len(tree.node_index) + 1  # +1 for goal state
    n_actions = len(interactive)

    # Transition matrix: from each interactive node, action leads to next
    T = np.zeros((n_states, n_states))
    for i, node in enumerate(interactive):
        src_idx = list(tree.node_index.keys()).index(node.id)
        # Deterministic transition to "next" state
        dst_idx = (src_idx + 1) % (n_states - 1)
        T[src_idx, dst_idx] = 0.9
        T[src_idx, src_idx] = 0.1  # chance of staying

    # Absorbing goal state
    T[n_states - 1, n_states - 1] = 1.0

    # Reward: -cost for each action, +1 for reaching goal
    R = np.full(n_states, -0.1)
    R[n_states - 1] = 1.0

    # Entropy rate of the transition matrix
    # Only use non-zero rows
    valid_rows = T.sum(axis=1) > 0
    T_valid = T[valid_rows][:, valid_rows] if np.sum(valid_rows) > 1 else T[:1, :1]
    h_rate = markov_entropy_rate(T_valid) if T_valid.shape[0] > 1 else 0.0

    mdp = {
        "n_states": n_states,
        "n_actions": n_actions,
        "transition_density": float(np.count_nonzero(T)) / max(T.size, 1),
        "entropy_rate": h_rate,
    }

    for k, v in mdp.items():
        print(f"    {k}: {v}")

    return mdp


# ---------------------------------------------------------------------------
# Stage 4: Solve — bounded-rational policy
# ---------------------------------------------------------------------------

def stage_solve(tree: AccessibilityTree, label: str) -> dict[str, float]:
    """Simulate a bounded-rational agent navigating the UI."""
    print(f"\n[Stage 4] Solving bounded-rational policy ({label})...")

    agent = BoundedRationalAgent(
        rationality=3.0,
        motor_noise=0.1,
        memory_capacity=5,
        seed=42,
    )
    env = UIEnvironment(tree)

    # Run the agent for a few episodes
    episode_costs = []
    episode_steps = []
    for ep in range(10):
        env.reset()
        total_cost = 0.0
        steps = 0
        for _ in range(50):
            actions = env.available_actions()
            if not actions:
                break
            action = agent.select_action(env.current_state(), actions)
            cost = agent.action_cost(action)
            total_cost += cost
            env.step(action)
            steps += 1
            if env.is_done():
                break
        episode_costs.append(total_cost)
        episode_steps.append(steps)

    metrics = {
        "mean_cost": float(np.mean(episode_costs)),
        "std_cost": float(np.std(episode_costs)),
        "mean_steps": float(np.mean(episode_steps)),
        "max_steps": float(np.max(episode_steps)),
    }

    for k, v in metrics.items():
        print(f"    {k}: {v:.3f}")

    return metrics


# ---------------------------------------------------------------------------
# Stage 5: Compare
# ---------------------------------------------------------------------------

def stage_compare(
    costs_before: dict[str, float],
    costs_after: dict[str, float],
    solve_before: dict[str, float],
    solve_after: dict[str, float],
) -> RegressionVerdict:
    """Compare before/after metrics to determine verdict."""
    print("\n[Stage 5] Comparing before → after...")

    cost_increase = costs_after["total_estimated_time"] - costs_before["total_estimated_time"]
    policy_degradation = solve_after["mean_cost"] - solve_before["mean_cost"]

    print(f"    Cost increase:       {cost_increase:+.3f}s")
    print(f"    Policy degradation:  {policy_degradation:+.3f}")

    if cost_increase > 0.5 or policy_degradation > 0.3:
        verdict = RegressionVerdict.REGRESSION
    elif cost_increase < -0.3 and policy_degradation < -0.2:
        verdict = RegressionVerdict.IMPROVEMENT
    else:
        verdict = RegressionVerdict.NEUTRAL

    print(f"    Verdict: {verdict.value}")
    return verdict


# ---------------------------------------------------------------------------
# Stage 6: Bottleneck identification
# ---------------------------------------------------------------------------

def stage_bottleneck(
    tree_after: AccessibilityTree,
    costs_after: dict[str, float],
) -> list[dict[str, Any]]:
    """Identify and rank usability bottlenecks."""
    print("\n[Stage 6] Identifying bottlenecks...")

    bottlenecks = []

    # Check label coverage
    interactive = [
        n for n in tree_after.node_index.values()
        if n.role.lower() in ("button", "textbox", "link", "checkbox")
    ]
    unlabeled = [n for n in interactive if not n.name or not n.name.strip()]
    if unlabeled:
        bottlenecks.append({
            "type": BottleneckType.PERCEPTUAL_OVERLOAD.value,
            "description": f"{len(unlabeled)} interactive elements lack labels",
            "severity": len(unlabeled) / max(len(interactive), 1),
            "affected_nodes": [n.id for n in unlabeled[:5]],
        })

    # Check target sizes
    small_targets = [
        n for n in interactive
        if n.bounding_box and n.bounding_box.width * n.bounding_box.height < 44 * 44
    ]
    if small_targets:
        bottlenecks.append({
            "type": BottleneckType.MOTOR_DIFFICULTY.value,
            "description": f"{len(small_targets)} targets below 44×44px minimum",
            "severity": len(small_targets) / max(len(interactive), 1),
            "affected_nodes": [n.id for n in small_targets[:5]],
        })

    # Check choice overload
    if len(interactive) > 7:  # Miller's law
        bottlenecks.append({
            "type": BottleneckType.CHOICE_PARALYSIS.value,
            "description": f"{len(interactive)} choices exceeds working memory (7±2)",
            "severity": min(1.0, (len(interactive) - 7) / 10),
            "affected_nodes": [],
        })

    for bn in bottlenecks:
        print(f"    [{bn['type']}] {bn['description']} (severity={bn['severity']:.2f})")

    return bottlenecks


# ---------------------------------------------------------------------------
# Stage 7: Report
# ---------------------------------------------------------------------------

def stage_report(
    verdict: RegressionVerdict,
    bottlenecks: list[dict[str, Any]],
    costs_before: dict[str, float],
    costs_after: dict[str, float],
) -> None:
    """Generate a text report."""
    print("\n[Stage 7] Generating report...")
    print()

    summary = generate_executive_summary(
        verdict=verdict.value,
        confidence=0.85,
        bottleneck_count=len(bottlenecks),
        cost_before=costs_before["total_estimated_time"],
        cost_after=costs_after["total_estimated_time"],
    )
    print(summary)


# ---------------------------------------------------------------------------
# Import needed for type annotations
# ---------------------------------------------------------------------------
from typing import Any


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("FULL PIPELINE DEMONSTRATION")
    print("=" * 60)

    # Stage 1
    before, after = stage_parse()

    # Stage 2
    costs_before = stage_cost(before, "before")
    costs_after = stage_cost(after, "after")

    # Stage 3
    mdp_before = stage_mdp(before, "before")
    mdp_after = stage_mdp(after, "after")

    # Stage 4
    solve_before = stage_solve(before, "before")
    solve_after = stage_solve(after, "after")

    # Stage 5
    verdict = stage_compare(costs_before, costs_after, solve_before, solve_after)

    # Stage 6
    bottlenecks = stage_bottleneck(after, costs_after)

    # Stage 7
    stage_report(verdict, bottlenecks, costs_before, costs_after)

    # Complexity scaling test
    print("\n" + "-" * 60)
    print("Bonus: Empirical complexity scaling")
    sizes = [5, 10, 20, 50]
    times = []
    gen = SyntheticUIGenerator(seed=0)
    import time
    for n in sizes:
        t0 = time.time()
        tree = gen.generate_form(n_fields=n)
        comparator = BaselineComparator(seed=0)
        comparator.run_all(tree, tree)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  n={n:3d} fields  →  {elapsed:.4f}s")

    if len(sizes) >= 3:
        result = fit_complexity_model(
            [float(s) for s in sizes],
            times,
        )
        print(f"\n  Best fit: {result['best_model']} (R²={result['best_r_squared']:.3f})")

    print("\n✅ Full pipeline demo complete.")


if __name__ == "__main__":
    main()
