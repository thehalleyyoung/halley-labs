#!/usr/bin/env python3
"""
Example: Custom task specification for targeted usability analysis.

Demonstrates how to:
1. Define custom task specifications with goal states.
2. Build a complex multi-step UI (checkout flow).
3. Simulate user navigation with a bounded-rational agent.
4. Measure task-specific metrics (time to complete, errors, etc.).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.core.enums import BottleneckType
from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.simulation.agent import BoundedRationalAgent
from usability_oracle.simulation.environment import UIEnvironment
from usability_oracle.simulation.interaction import InteractionEvent, InteractionSequence
from usability_oracle.analysis.statistical import cohens_d, bootstrap_ci


# ---------------------------------------------------------------------------
# Task specification
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """A goal-directed task specification."""

    name: str
    description: str
    goal_node_ids: list[str] = field(default_factory=list)
    required_actions: list[str] = field(default_factory=list)
    max_allowed_steps: int = 50
    success_criteria: dict[str, Any] = field(default_factory=dict)

    def is_goal_reached(self, visited_nodes: set[str]) -> bool:
        """Check if all goal nodes have been visited."""
        return all(g in visited_nodes for g in self.goal_node_ids)


# ---------------------------------------------------------------------------
# Build checkout flow
# ---------------------------------------------------------------------------

def build_checkout_flow() -> AccessibilityTree:
    """Build a multi-step checkout form."""
    root = AccessibilityNode(
        id="root", role="main", name="Checkout",
        bounding_box=BoundingBox(x=0, y=0, width=600, height=1200),
        properties={}, state=AccessibilityState(), children=[], depth=0,
    )

    idx: dict[str, AccessibilityNode] = {"root": root}

    # Step 1: Shipping address
    sections = [
        ("shipping", "Shipping Address", [
            ("ship-name", "textbox", "Full Name"),
            ("ship-addr", "textbox", "Street Address"),
            ("ship-city", "textbox", "City"),
            ("ship-state", "combobox", "State"),
            ("ship-zip", "textbox", "ZIP Code"),
        ]),
        ("payment", "Payment Method", [
            ("pay-card", "textbox", "Card Number"),
            ("pay-exp", "textbox", "Expiry (MM/YY)"),
            ("pay-cvv", "textbox", "CVV"),
            ("pay-name", "textbox", "Cardholder Name"),
        ]),
        ("review", "Review Order", [
            ("review-summary", "text", "Order Summary"),
            ("review-total", "text", "Total: $99.99"),
            ("review-agree", "checkbox", "I agree to terms"),
        ]),
    ]

    y_pos = 10
    for section_id, section_name, fields in sections:
        section = AccessibilityNode(
            id=section_id, role="group", name=section_name,
            bounding_box=BoundingBox(x=10, y=y_pos, width=580, height=len(fields) * 50 + 40),
            properties={}, state=AccessibilityState(), children=[], depth=1,
            parent_id="root",
        )
        idx[section_id] = section

        heading = AccessibilityNode(
            id=f"{section_id}-heading", role="heading", name=section_name,
            bounding_box=BoundingBox(x=20, y=y_pos + 5, width=560, height=30),
            properties={"level": "2"}, state=AccessibilityState(), children=[],
            depth=2, parent_id=section_id,
        )
        section.children.append(heading)
        idx[heading.id] = heading
        y_pos += 40

        for fid, frole, fname in fields:
            label = AccessibilityNode(
                id=f"{fid}-label", role="label", name=fname,
                bounding_box=BoundingBox(x=20, y=y_pos, width=150, height=20),
                properties={"for": fid}, state=AccessibilityState(), children=[],
                depth=2, parent_id=section_id,
            )
            field_node = AccessibilityNode(
                id=fid, role=frole, name=fname,
                bounding_box=BoundingBox(x=180, y=y_pos, width=400, height=36),
                properties={"aria-required": "true"},
                state=AccessibilityState(), children=[], depth=2,
                parent_id=section_id,
            )
            section.children.extend([label, field_node])
            idx[label.id] = label
            idx[fid] = field_node
            y_pos += 50

        root.children.append(section)
        y_pos += 20

    # Place order button
    submit = AccessibilityNode(
        id="place-order", role="button", name="Place Order",
        bounding_box=BoundingBox(x=200, y=y_pos, width=200, height=48),
        properties={}, state=AccessibilityState(), children=[], depth=1,
        parent_id="root",
    )
    root.children.append(submit)
    idx[submit.id] = submit

    return AccessibilityTree(root=root, node_index=idx)


# ---------------------------------------------------------------------------
# Define tasks
# ---------------------------------------------------------------------------

def define_tasks() -> list[TaskSpec]:
    """Define several checkout-related tasks."""
    return [
        TaskSpec(
            name="complete_checkout",
            description="Fill all fields and place order",
            goal_node_ids=["place-order"],
            required_actions=["ship-name", "ship-addr", "ship-city",
                            "ship-state", "ship-zip", "pay-card",
                            "pay-exp", "pay-cvv", "pay-name",
                            "review-agree", "place-order"],
            max_allowed_steps=30,
        ),
        TaskSpec(
            name="shipping_only",
            description="Fill only the shipping section",
            goal_node_ids=["ship-zip"],
            required_actions=["ship-name", "ship-addr", "ship-city",
                            "ship-state", "ship-zip"],
            max_allowed_steps=15,
        ),
        TaskSpec(
            name="review_and_submit",
            description="Agree to terms and place order (assume fields filled)",
            goal_node_ids=["place-order"],
            required_actions=["review-agree", "place-order"],
            max_allowed_steps=10,
        ),
    ]


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

def simulate_task(
    tree: AccessibilityTree,
    task: TaskSpec,
    agent: BoundedRationalAgent,
    n_episodes: int = 20,
) -> dict[str, Any]:
    """Simulate a bounded-rational agent performing a task."""
    env = UIEnvironment(tree)

    episode_metrics: list[dict[str, float]] = []
    for ep in range(n_episodes):
        env.reset()
        visited: set[str] = set()
        total_cost = 0.0
        steps = 0
        errors = 0

        for step in range(task.max_allowed_steps):
            actions = env.available_actions()
            if not actions:
                break

            action = agent.select_action(env.current_state(), actions)
            cost = agent.action_cost(action)
            total_cost += cost
            steps += 1

            # Track visited nodes
            if hasattr(action, "target_id"):
                visited.add(action.target_id)
            elif hasattr(action, "node_id"):
                visited.add(action.node_id)

            env.step(action)

            # Check if task action was required
            if hasattr(action, "target_id") and action.target_id not in task.required_actions:
                errors += 1

            if task.is_goal_reached(visited):
                break

        success = task.is_goal_reached(visited)
        episode_metrics.append({
            "success": 1.0 if success else 0.0,
            "steps": float(steps),
            "total_cost": total_cost,
            "errors": float(errors),
            "completion_rate": len(visited & set(task.required_actions)) / max(len(task.required_actions), 1),
        })

    # Aggregate
    successes = [m["success"] for m in episode_metrics]
    costs = [m["total_cost"] for m in episode_metrics]
    step_counts = [m["steps"] for m in episode_metrics]
    error_counts = [m["errors"] for m in episode_metrics]
    completions = [m["completion_rate"] for m in episode_metrics]

    return {
        "task": task.name,
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(successes)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "mean_steps": float(np.mean(step_counts)),
        "mean_errors": float(np.mean(error_counts)),
        "mean_completion": float(np.mean(completions)),
        "cost_ci": bootstrap_ci(costs, confidence=0.95),
    }


# ---------------------------------------------------------------------------
# Compare agent rationality levels
# ---------------------------------------------------------------------------

def compare_rationality_levels(
    tree: AccessibilityTree,
    task: TaskSpec,
    levels: list[float] = [0.5, 1.0, 3.0, 10.0],
    n_episodes: int = 15,
) -> list[dict[str, Any]]:
    """Compare task performance at different rationality levels."""
    results = []
    for beta in levels:
        agent = BoundedRationalAgent(
            rationality=beta,
            motor_noise=0.1,
            memory_capacity=5,
            seed=42,
        )
        metrics = simulate_task(tree, task, agent, n_episodes)
        metrics["rationality"] = beta
        results.append(metrics)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("CUSTOM TASK SPECIFICATION EXAMPLE")
    print("=" * 60)

    # Build checkout flow
    tree = build_checkout_flow()
    print(f"\nCheckout flow: {len(tree.node_index)} nodes")

    # Define tasks
    tasks = define_tasks()
    print(f"Defined {len(tasks)} tasks")

    # Run each task
    agent = BoundedRationalAgent(
        rationality=3.0,
        motor_noise=0.1,
        memory_capacity=5,
        seed=42,
    )

    print("\n" + "-" * 60)
    print("Task Performance (β=3.0):")
    for task in tasks:
        metrics = simulate_task(tree, task, agent, n_episodes=10)
        print(f"\n  Task: {task.name}")
        print(f"    Success rate:  {metrics['success_rate']:.0%}")
        print(f"    Mean cost:     {metrics['mean_cost']:.3f}")
        print(f"    Mean steps:    {metrics['mean_steps']:.1f}")
        print(f"    Mean errors:   {metrics['mean_errors']:.1f}")
        print(f"    Completion:    {metrics['mean_completion']:.0%}")
        print(f"    Cost 95% CI:   {metrics['cost_ci']}")

    # Compare rationality levels
    print("\n" + "-" * 60)
    print("Rationality Level Comparison (complete_checkout task):")
    comparison = compare_rationality_levels(tree, tasks[0], n_episodes=10)
    for r in comparison:
        print(f"  β={r['rationality']:<5.1f}  success={r['success_rate']:.0%}  "
              f"cost={r['mean_cost']:.3f}  steps={r['mean_steps']:.1f}")

    # Effect size between lowest and highest rationality
    if len(comparison) >= 2:
        costs_low = [0.0] * 10  # placeholder for aggregation
        costs_high = [0.0] * 10
        d = cohens_d(
            [comparison[0]["mean_cost"]] * 10,
            [comparison[-1]["mean_cost"]] * 10,
        )
        print(f"\n  Cohen's d (lowest vs highest β): {d:.3f}")

    print("\n✅ Custom task specification example complete.")


if __name__ == "__main__":
    main()
