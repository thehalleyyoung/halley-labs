"""Paper-formatted experiment results."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_experiments import (
    experiment_1_selection_methods,
    experiment_2_adaptive_kernel,
    experiment_3_convergence,
    experiment_4_coverage_high_dim,
    experiment_5_pareto_frontier,
    experiment_6_scoring_properness,
    experiment_7_diversity_objectives,
    experiment_8_quality_metrics,
    experiment_9_mechanism_ic,
    experiment_10_significance_tests,
)


def format_table(title, headers, rows):
    """Format a table for paper output."""
    print(f"\n{'=' * 70}")
    print(f"Table: {title}")
    print(f"{'=' * 70}")
    col_widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def main():
    print("Running paper experiments...")
    print("=" * 70)

    # Experiment 1
    r1 = experiment_1_selection_methods()
    rows = []
    for method in ["random", "top_quality", "dpp_greedy", "flow", "vcg", "budget_feasible"]:
        d = r1[method]
        rows.append([method,
                     f"{d['cosine_div']['mean']:.4f}±{d['cosine_div']['std']:.4f}",
                     f"{d['dispersion']['mean']:.4f}",
                     f"{d['mean_quality']['mean']:.4f}",
                     f"{d['ic_violations']['mean']:.1f}"])
    format_table("Selection Method Comparison (N=200, k=10, 5 seeds)",
                 ["Method", "Cosine Div", "Dispersion", "Quality", "IC Violations"], rows)

    # Experiment 2
    r2 = experiment_2_adaptive_kernel()
    rows = []
    for name, d in r2.items():
        rows.append([name, f"{d['cosine_div']['mean']:.4f}±{d['cosine_div']['std']:.4f}",
                     f"{d['dispersion']['mean']:.4f}"])
    format_table("Kernel Comparison (N=100, 5 seeds)", ["Kernel", "Cosine Div", "Dispersion"], rows)

    # Experiment 3
    r3 = experiment_3_convergence()
    rows = []
    for T in ["1", "2", "4", "8"]:
        if T in r3["flow"]:
            rows.append([T,
                        f"{r3['flow'][T]['diversity']['mean']:.4f}±{r3['flow'][T]['diversity']['std']:.4f}",
                        f"{r3['flow'][T]['regret']['mean']:.4f}±{r3['flow'][T]['regret']['std']:.4f}"])
    format_table("Flow Convergence Analysis (5 seeds)",
                 ["Rounds", "Flow Diversity", "Flow Regret"], rows)

    # Experiment 8: Quality Metrics
    r8 = experiment_8_quality_metrics()
    rows = []
    for comp in ["coherence", "relevance", "fluency", "consistency", "aggregate"]:
        s = r8["component_statistics"][comp]
        rows.append([comp, f"{s['mean']:.4f}±{s['std']:.4f}"])
    format_table("Multi-Dimensional Quality Metrics", ["Component", "Score"], rows)

    # Experiment 9: IC Verification
    r9 = experiment_9_mechanism_ic()
    rows = []
    for mech_name in ["direct", "vcg", "budget_feasible", "mmr", "kmedoids"]:
        d = r9[mech_name]
        rows.append([mech_name,
                     f"{d['ic_violations']['mean']:.1f}±{d['ic_violations']['std']:.1f}",
                     f"{d['diversity']['mean']:.4f}",
                     f"{d['total_payment']['mean']:.4f}"])
    format_table("Mechanism IC Verification (5 seeds)",
                 ["Mechanism", "IC Violations", "Diversity", "Total Payment"], rows)

    # Experiment 10: Significance
    r10 = experiment_10_significance_tests()
    rows = []
    for comp_name, d in r10.items():
        rows.append([comp_name,
                     f"{d['t_statistic']:.4f}",
                     f"{d['p_value']:.6f}",
                     str(d.get('significant_at_005', ''))])
    format_table("Statistical Significance (paired t-tests)",
                 ["Comparison", "t-stat", "p-value", "Sig (α=0.05)"], rows)

    print("\n" + "=" * 70)
    print("All paper experiments complete.")


if __name__ == "__main__":
    main()
