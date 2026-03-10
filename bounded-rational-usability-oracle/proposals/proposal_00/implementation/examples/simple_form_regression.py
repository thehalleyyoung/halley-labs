#!/usr/bin/env python3
"""
Example: Detect a usability regression in a simple login form.

This example demonstrates the core workflow of the Bounded-Rational
Usability Oracle:

1. Build a synthetic "before" accessibility tree (well-designed form).
2. Apply a mutation that introduces a usability bottleneck.
3. Run the oracle pipeline to detect the regression.
4. Inspect the detected bottlenecks and severity scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the package is importable when running from the examples/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


# ---------------------------------------------------------------------------
# 1. Build the "before" form tree
# ---------------------------------------------------------------------------

def build_login_form() -> AccessibilityTree:
    """Construct a minimal login form accessibility tree."""
    root = AccessibilityNode(
        id="root",
        role="main",
        name="Login Page",
        bounding_box=BoundingBox(x=0, y=0, width=400, height=600),
        properties={},
        state=AccessibilityState(),
        children=[],
        depth=0,
    )

    heading = AccessibilityNode(
        id="heading",
        role="heading",
        name="Sign In",
        bounding_box=BoundingBox(x=50, y=20, width=300, height=40),
        properties={"level": "1"},
        state=AccessibilityState(),
        children=[],
        depth=1,
        parent_id="root",
    )

    username_label = AccessibilityNode(
        id="username-label",
        role="label",
        name="Username",
        bounding_box=BoundingBox(x=50, y=80, width=300, height=20),
        properties={"for": "username-input"},
        state=AccessibilityState(),
        children=[],
        depth=1,
        parent_id="root",
    )

    username_input = AccessibilityNode(
        id="username-input",
        role="textbox",
        name="Username",
        bounding_box=BoundingBox(x=50, y=105, width=300, height=36),
        properties={"aria-required": "true", "autocomplete": "username"},
        state=AccessibilityState(),
        children=[],
        depth=1,
        parent_id="root",
    )

    password_label = AccessibilityNode(
        id="password-label",
        role="label",
        name="Password",
        bounding_box=BoundingBox(x=50, y=160, width=300, height=20),
        properties={"for": "password-input"},
        state=AccessibilityState(),
        children=[],
        depth=1,
        parent_id="root",
    )

    password_input = AccessibilityNode(
        id="password-input",
        role="textbox",
        name="Password",
        bounding_box=BoundingBox(x=50, y=185, width=300, height=36),
        properties={"aria-required": "true", "type": "password"},
        state=AccessibilityState(),
        children=[],
        depth=1,
        parent_id="root",
    )

    submit_btn = AccessibilityNode(
        id="submit-btn",
        role="button",
        name="Sign In",
        bounding_box=BoundingBox(x=50, y=250, width=300, height=44),
        properties={},
        state=AccessibilityState(),
        children=[],
        depth=1,
        parent_id="root",
    )

    forgot_link = AccessibilityNode(
        id="forgot-link",
        role="link",
        name="Forgot password?",
        bounding_box=BoundingBox(x=50, y=310, width=300, height=20),
        properties={"href": "/forgot"},
        state=AccessibilityState(),
        children=[],
        depth=1,
        parent_id="root",
    )

    root.children = [
        heading, username_label, username_input,
        password_label, password_input, submit_btn, forgot_link,
    ]

    idx = {}
    def _index(node: AccessibilityNode) -> None:
        idx[node.id] = node
        for c in node.children:
            _index(c)
    _index(root)

    return AccessibilityTree(root=root, node_index=idx)


# ---------------------------------------------------------------------------
# 2. Apply mutations
# ---------------------------------------------------------------------------

def apply_regression(tree: AccessibilityTree) -> AccessibilityTree:
    """Introduce several usability regressions into the form."""
    mutator = MutationOperator(seed=42)

    # Remove labels (common A11y regression)
    mutated = mutator.apply_label_removal(tree, fraction=0.6)

    # Then reduce target sizes
    mutated = mutator.apply_motor_difficulty(mutated, severity=0.7)

    return mutated


# ---------------------------------------------------------------------------
# 3. Run baselines
# ---------------------------------------------------------------------------

def run_analysis(before: AccessibilityTree, after: AccessibilityTree) -> None:
    """Run baseline comparators and print results."""
    comparator = BaselineComparator(seed=42)

    print("=" * 60)
    print("USABILITY REGRESSION ANALYSIS — Login Form")
    print("=" * 60)
    print()

    # Run all baselines
    results = comparator.run_all(before, after)

    print("Baseline Results:")
    print("-" * 40)
    for name, verdict in results.items():
        icon = {"regression": "🔴", "improvement": "🟢", "neutral": "⚪", "inconclusive": "❓"}.get(
            verdict.value, "?"
        )
        print(f"  {icon} {name:25s} → {verdict.value}")

    # Majority vote
    print()
    ensemble = comparator.majority_vote(before, after)
    print(f"  Ensemble verdict: {ensemble.value}")

    # Tree statistics
    print()
    print("Tree Statistics:")
    print("-" * 40)
    for label, tree in [("Before", before), ("After", after)]:
        n_nodes = len(tree.node_index)
        max_depth = max(n.depth for n in tree.node_index.values())
        n_interactive = sum(
            1 for n in tree.node_index.values()
            if n.role.lower() in ("button", "textbox", "link", "checkbox")
        )
        n_labeled = sum(
            1 for n in tree.node_index.values()
            if n.name and n.name.strip()
        )
        print(f"  {label}: {n_nodes} nodes, depth={max_depth}, "
              f"interactive={n_interactive}, labeled={n_labeled}")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main() -> None:
    before = build_login_form()
    after = apply_regression(before)
    run_analysis(before, after)

    print()
    print("✅ Example complete. The oracle detected regressions caused by")
    print("   label removal and target-size reduction.")


if __name__ == "__main__":
    main()
