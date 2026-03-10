#!/usr/bin/env python3
"""
Example: Dashboard fragility analysis.

Demonstrates how to:
1. Generate a synthetic dashboard with multiple widget types.
2. Apply a suite of mutation operators at varying severity levels.
3. Measure the fragility (sensitivity to perturbation) of the UI.
4. Produce a fragility report with per-bottleneck-type breakdown.
"""

from __future__ import annotations

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
from usability_oracle.benchmarks.mutations import MutationOperator
from usability_oracle.evaluation.baselines import BaselineComparator
from usability_oracle.core.enums import BottleneckType, RegressionVerdict
from usability_oracle.analysis.sensitivity import (
    morris_elementary_effects,
    local_sensitivity,
)
from usability_oracle.analysis.statistical import (
    cohens_d,
    mann_whitney_u,
)


# ---------------------------------------------------------------------------
# Build dashboard tree
# ---------------------------------------------------------------------------

def build_dashboard() -> AccessibilityTree:
    """Create a synthetic dashboard with charts, tables, and controls."""
    gen = SyntheticUIGenerator(seed=42)
    # Use the form generator as a base and extend it
    tree = gen.generate_form(n_fields=8, include_validation=True)

    # Add some extra "chart" and "status" widgets manually
    root = tree.root
    y_offset = 500  # below the form

    for i in range(4):
        chart = AccessibilityNode(
            id=f"chart-{i}",
            role="img",
            name=f"Chart: Metric {i + 1}",
            bounding_box=BoundingBox(
                x=(i % 2) * 300, y=y_offset + (i // 2) * 200,
                width=280, height=180,
            ),
            properties={"aria-label": f"Bar chart showing metric {i + 1} trends"},
            state=AccessibilityState(),
            children=[],
            depth=1,
            parent_id="root",
        )
        root.children.append(chart)
        tree.node_index[chart.id] = chart

    # Status indicators
    for i in range(3):
        status = AccessibilityNode(
            id=f"status-{i}",
            role="status",
            name=f"System {i + 1}: OK",
            bounding_box=BoundingBox(x=10 + i * 200, y=y_offset + 420, width=180, height=30),
            properties={"aria-live": "polite"},
            state=AccessibilityState(),
            children=[],
            depth=1,
            parent_id="root",
        )
        root.children.append(status)
        tree.node_index[status.id] = status

    return tree


# ---------------------------------------------------------------------------
# Fragility sweep
# ---------------------------------------------------------------------------

def fragility_sweep(
    original: AccessibilityTree,
    n_severities: int = 10,
    n_repeats: int = 5,
) -> dict[str, list[dict[str, float]]]:
    """Apply each mutation type at increasing severity and measure impact."""
    mutator = MutationOperator(seed=42)
    comparator = BaselineComparator(seed=42)

    mutation_fns = [
        ("perceptual_overload", mutator.apply_perceptual_overload),
        ("choice_paralysis", mutator.apply_choice_paralysis),
        ("motor_difficulty", mutator.apply_motor_difficulty),
        ("memory_decay", mutator.apply_memory_decay),
        ("interference", mutator.apply_interference),
        ("label_removal", mutator.apply_label_removal),
        ("contrast_reduction", mutator.apply_contrast_reduction),
    ]

    severities = np.linspace(0.1, 0.9, n_severities)
    results: dict[str, list[dict[str, float]]] = {}

    for name, fn in mutation_fns:
        results[name] = []
        for sev in severities:
            regression_count = 0
            for rep in range(n_repeats):
                try:
                    mutated = fn(original, float(sev))
                    verdicts = comparator.run_all(original, mutated)
                    n_regressions = sum(
                        1 for v in verdicts.values() if v == RegressionVerdict.REGRESSION
                    )
                    regression_count += n_regressions
                except Exception:
                    regression_count += 0

            avg_regressions = regression_count / n_repeats
            n_baselines = len(comparator.run_all(original, original))
            fragility = avg_regressions / n_baselines if n_baselines > 0 else 0.0

            results[name].append({
                "severity": float(sev),
                "avg_regressions": avg_regressions,
                "fragility": fragility,
            })

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_fragility_report(results: dict[str, list[dict[str, float]]]) -> None:
    """Print a text-based fragility report."""
    print("=" * 70)
    print("DASHBOARD FRAGILITY REPORT")
    print("=" * 70)

    for mutation_name, sweep in results.items():
        print(f"\n  {mutation_name}:")

        # ASCII bar chart
        max_frag = max(r["fragility"] for r in sweep) if sweep else 1.0
        scale = 40.0 / max(max_frag, 0.01)

        for r in sweep:
            bar_len = int(r["fragility"] * scale)
            bar = "█" * bar_len
            print(f"    sev={r['severity']:.2f}  {bar:40s}  {r['fragility']:.3f}")

    # Summary: which mutation type is most dangerous?
    print("\n" + "-" * 70)
    print("  Mutation Impact Summary (avg fragility across severities):")
    summaries = []
    for name, sweep in results.items():
        avg_frag = np.mean([r["fragility"] for r in sweep])
        max_frag = max(r["fragility"] for r in sweep)
        summaries.append((name, avg_frag, max_frag))

    summaries.sort(key=lambda x: x[1], reverse=True)
    for name, avg_f, max_f in summaries:
        print(f"    {name:25s}  avg={avg_f:.3f}  max={max_f:.3f}")


# ---------------------------------------------------------------------------
# Statistical tests on fragility
# ---------------------------------------------------------------------------

def statistical_analysis(results: dict[str, list[dict[str, float]]]) -> None:
    """Run pairwise statistical comparisons between mutation types."""
    print("\n" + "-" * 70)
    print("  Pairwise Comparisons (Cohen's d):")

    names = list(results.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vals_i = [r["fragility"] for r in results[names[i]]]
            vals_j = [r["fragility"] for r in results[names[j]]]
            d = cohens_d(vals_i, vals_j)
            significance = "***" if abs(d) > 0.8 else "**" if abs(d) > 0.5 else "*" if abs(d) > 0.2 else ""
            print(f"    {names[i]:20s} vs {names[j]:20s}  d={d:+.3f} {significance}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building dashboard...")
    dashboard = build_dashboard()
    n_nodes = len(dashboard.node_index)
    print(f"  Dashboard has {n_nodes} nodes")

    print("\nRunning fragility sweep (this may take a moment)...")
    results = fragility_sweep(dashboard, n_severities=5, n_repeats=3)

    print_fragility_report(results)
    statistical_analysis(results)

    print("\n✅ Dashboard fragility analysis complete.")


if __name__ == "__main__":
    main()
