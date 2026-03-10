#!/usr/bin/env python3
"""
Example: Analyse a navigation menu for usability bottlenecks.

Demonstrates:
- Generating synthetic navigation menus of varying complexity.
- Computing Hick-Hyman information-theoretic costs.
- Measuring cognitive load via entropy analysis.
- Comparing shallow vs deep menu structures.
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
from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.utils.entropy import (
    entropy,
    effective_number,
    conditional_entropy,
    kl_divergence,
    jensen_shannon_divergence,
    renyi_entropy,
)
from usability_oracle.evaluation.baselines import BaselineComparator


# ---------------------------------------------------------------------------
# Build navigation trees
# ---------------------------------------------------------------------------

def build_flat_menu(n_items: int = 12) -> AccessibilityTree:
    """Build a single-level navigation with *n_items* links."""
    root = AccessibilityNode(
        id="root", role="navigation", name="Main Menu",
        bounding_box=BoundingBox(x=0, y=0, width=200, height=n_items * 40),
        properties={}, state=AccessibilityState(), children=[], depth=0,
    )

    for i in range(n_items):
        link = AccessibilityNode(
            id=f"link-{i}", role="link", name=f"Section {i + 1}",
            bounding_box=BoundingBox(x=0, y=i * 40, width=200, height=36),
            properties={"href": f"/section-{i + 1}"},
            state=AccessibilityState(), children=[], depth=1, parent_id="root",
        )
        root.children.append(link)

    idx = {n.id: n for n in [root] + root.children}
    return AccessibilityTree(root=root, node_index=idx)


def build_hierarchical_menu(
    categories: int = 4,
    items_per_category: int = 3,
) -> AccessibilityTree:
    """Build a two-level navigation with expandable categories."""
    total_items = categories * items_per_category
    root = AccessibilityNode(
        id="root", role="navigation", name="Main Menu",
        bounding_box=BoundingBox(x=0, y=0, width=200, height=categories * 40 + total_items * 30),
        properties={}, state=AccessibilityState(), children=[], depth=0,
    )

    idx: dict[str, AccessibilityNode] = {"root": root}
    y_pos = 0

    for c in range(categories):
        group = AccessibilityNode(
            id=f"category-{c}", role="group", name=f"Category {c + 1}",
            bounding_box=BoundingBox(x=0, y=y_pos, width=200, height=40),
            properties={"aria-expanded": "true"},
            state=AccessibilityState(expanded=True), children=[], depth=1, parent_id="root",
        )
        y_pos += 40
        idx[group.id] = group

        for i in range(items_per_category):
            link = AccessibilityNode(
                id=f"link-{c}-{i}", role="link",
                name=f"Category {c + 1} / Item {i + 1}",
                bounding_box=BoundingBox(x=20, y=y_pos, width=180, height=28),
                properties={"href": f"/cat-{c}/item-{i}"},
                state=AccessibilityState(), children=[], depth=2,
                parent_id=group.id,
            )
            group.children.append(link)
            idx[link.id] = link
            y_pos += 30

        root.children.append(group)

    return AccessibilityTree(root=root, node_index=idx)


# ---------------------------------------------------------------------------
# Hick-Hyman analysis
# ---------------------------------------------------------------------------

def hick_hyman_time(n_choices: int, base_time: float = 0.2) -> float:
    """Hick-Hyman law: T = a + b * log₂(n + 1)."""
    return base_time + 0.15 * math.log2(n_choices + 1)


def analyse_menu_costs(tree: AccessibilityTree, label: str) -> dict[str, float]:
    """Compute information-theoretic cost metrics for a menu."""
    # Count interactive items
    interactive = [
        n for n in tree.node_index.values()
        if n.role.lower() in ("link", "button", "menuitem")
    ]
    n_items = len(interactive)

    # Uniform selection probability
    if n_items > 0:
        probs = np.ones(n_items) / n_items
    else:
        probs = np.array([1.0])

    # Compute metrics
    h = entropy(probs)
    eff_n = effective_number(probs)
    hick_time = hick_hyman_time(n_items)
    renyi_2 = renyi_entropy(probs, alpha=2.0)

    # Categorised items (by depth)
    depth_counts: dict[int, int] = {}
    for n in interactive:
        depth_counts[n.depth] = depth_counts.get(n.depth, 0) + 1

    metrics = {
        "label": label,
        "n_items": n_items,
        "entropy_bits": h,
        "effective_choices": eff_n,
        "hick_hyman_time_s": hick_time,
        "renyi_entropy_2": renyi_2,
        "max_depth": max((n.depth for n in tree.node_index.values()), default=0),
        "depth_distribution": depth_counts,
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("NAVIGATION MENU ANALYSIS")
    print("=" * 60)

    # Build two menu variants
    flat = build_flat_menu(n_items=12)
    hierarchical = build_hierarchical_menu(categories=4, items_per_category=3)

    for tree, label in [(flat, "Flat (12 items)"), (hierarchical, "Hierarchical (4×3)")]:
        metrics = analyse_menu_costs(tree, label)
        print(f"\n{label}:")
        print(f"  Items:           {metrics['n_items']}")
        print(f"  Entropy:         {metrics['entropy_bits']:.3f} bits")
        print(f"  Effective N:     {metrics['effective_choices']:.1f}")
        print(f"  Hick-Hyman:      {metrics['hick_hyman_time_s']:.3f} s")
        print(f"  Rényi H₂:        {metrics['renyi_entropy_2']:.3f}")
        print(f"  Max depth:       {metrics['max_depth']}")
        print(f"  Depth dist:      {metrics['depth_distribution']}")

    # Compare using baselines
    print("\n" + "-" * 60)
    print("Baseline Comparison (flat → hierarchical):")
    comparator = BaselineComparator(seed=42)
    results = comparator.run_all(flat, hierarchical)
    for name, verdict in results.items():
        print(f"  {name:25s} → {verdict.value}")

    # Information divergence between menu distributions
    p_flat = np.ones(12) / 12
    # Hierarchical: user first picks category (4 options), then item (3 options)
    p_hier = np.ones(12) / 12  # same items, different structure
    jsd = jensen_shannon_divergence(p_flat, p_hier)
    print(f"\n  JS divergence (flat vs hier): {jsd:.6f}")

    # Demonstrate effect of increasing menu size
    print("\n" + "-" * 60)
    print("Scaling: Hick-Hyman time vs menu size")
    for n in [4, 8, 16, 32, 64]:
        t = hick_hyman_time(n)
        h = math.log2(n)
        print(f"  n={n:3d}  →  T={t:.3f}s  H={h:.2f} bits")

    print("\n✅ Navigation analysis complete.")


if __name__ == "__main__":
    main()
