#!/usr/bin/env python3
"""
CausalBound — DebtRank Analysis Example
========================================

This example demonstrates financial contagion modelling with the
``causalbound.contagion`` module.  We compare two contagion mechanisms:

    * **DebtRank** — a fixed-point iterative model that propagates
      continuous distress levels through the interbank liability matrix.
    * **Cascade** — a threshold-based default cascade where an institution
      fails when accumulated losses exceed its capital buffer.

The script generates a scale-free interbank network (which mimics the
heavy-tailed degree distributions observed in real OTC markets), applies
several shock scenarios, and reports:

    1. System-wide DebtRank for increasing shock severity
    2. Per-node distress propagation over rounds
    3. Sensitivity analysis to identify systemically important nodes
    4. Cascade model comparison with tipping-point detection
    5. Contagion-path analysis for the worst shock scenario

Usage
-----
    python debtrank_analysis.py
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

# ── CausalBound imports ──────────────────────────────────────────────────
from causalbound.network.generators import (
    ExposureParams,
    ScaleFreeGenerator,
)
from causalbound.network.topology import NetworkTopology, CentralityMethod
from causalbound.contagion.debtrank import (
    DebtRankModel,
    DebtRankVariant,
)
from causalbound.contagion.cascade import CascadeModel

# ── Reproducibility ──────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# =========================================================================
# Helper utilities
# =========================================================================

def separator_line(title: str, width: int = 72) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def bar_chart(values: Dict[int, float], label: str = "value",
              max_bar: int = 40) -> None:
    """Print a simple horizontal bar chart in the terminal."""
    if not values:
        return
    max_val = max(abs(v) for v in values.values()) or 1.0
    for node_id in sorted(values, key=lambda k: -values[k])[:15]:
        val = values[node_id]
        bar_len = int(abs(val) / max_val * max_bar)
        bar = "█" * bar_len
        print(f"    Node {node_id:3d} │ {bar} {val:.4f}")


# =========================================================================
# 1. Generate a scale-free financial network
# =========================================================================

def generate_network(n_nodes: int = 30) -> nx.DiGraph:
    """Build a scale-free interbank network.

    Scale-free networks exhibit the "too-big-to-fail" property naturally:
    a small number of hub nodes have very high degree while the majority
    of peripheral institutions maintain few connections.
    """
    separator_line("1. Generate Scale-Free Financial Network")

    exposure_params = ExposureParams(
        distribution="pareto",
        location=1e6,
        scale=1e8,
        shape=1.5,
        min_exposure=1e5,
        max_exposure=5e10,
    )
    generator = ScaleFreeGenerator(exposure_params=exposure_params, seed=SEED)

    G = generator.generate(
        n_nodes=n_nodes,
        m=3,               # each new node attaches to 3 existing nodes
        alpha=1.0,          # full preferential attachment
        reciprocity=0.5,
    )

    # Print basic statistics
    topo = NetworkTopology()
    report = topo.analyze(G)
    print(f"  Nodes            : {report.n_nodes}")
    print(f"  Edges            : {report.n_edges}")
    print(f"  Density          : {report.density:.4f}")
    print(f"  Reciprocity      : {report.reciprocity:.4f}")
    print(f"  Avg in-degree    : {report.degree_distribution.mean_in:.2f}")
    print(f"  Max in-degree    : {report.degree_distribution.max_in}")
    print(f"  Max out-degree   : {report.degree_distribution.max_out}")

    # Identify the top-5 nodes by PageRank
    pagerank = topo.get_centrality(G, method=CentralityMethod.PAGERANK)
    top5 = sorted(pagerank, key=lambda n: -pagerank[n])[:5]
    print(f"\n  Top-5 by PageRank:")
    for rank, node in enumerate(top5, 1):
        print(f"    {rank}. Node {node} — PageRank = {pagerank[node]:.4f}")

    return G


# =========================================================================
# 2. Run DebtRank with increasing shock severity
# =========================================================================

def run_debtrank_sweep(G: nx.DiGraph) -> List[Dict]:
    """Apply shocks of increasing magnitude to the most central node.

    This produces a "stress curve" showing how system-wide distress
    amplifies with the severity of an initial shock.
    """
    separator_line("2. DebtRank Shock Severity Sweep")

    model = DebtRankModel(
        variant=DebtRankVariant.LINEAR,
        default_threshold=1.0,
    )

    # Pick the highest-degree node as the shock target
    degrees = dict(G.degree())
    hub_node = max(degrees, key=lambda n: degrees[n])
    print(f"  Hub node selected: {hub_node} (degree {degrees[hub_node]})")

    shock_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = []

    print(f"\n  {'Shock':>8s}  {'SysRisk':>10s}  {'Cascade':>10s}  "
          f"{'Rounds':>8s}  {'Loss%':>8s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for shock in shock_levels:
        initial_shocks = {hub_node: shock}
        dr_result = model.compute(
            G, initial_shocks,
            max_rounds=100,
            track_history=True,
        )
        results.append({
            "shock": shock,
            "debtrank": dr_result.system_debtrank,
            "cascade_size": dr_result.cascade_size,
            "rounds": dr_result.rounds_propagated,
            "loss_frac": dr_result.loss_fraction,
            "result": dr_result,
        })
        print(f"  {shock:8.2f}  {dr_result.system_debtrank:10.6f}  "
              f"{dr_result.cascade_size:10d}  "
              f"{dr_result.rounds_propagated:8d}  "
              f"{dr_result.loss_fraction:8.4f}")

    return results


# =========================================================================
# 3. Multi-node shock analysis
# =========================================================================

def run_multinode_shocks(G: nx.DiGraph) -> Dict[str, float]:
    """Compare the impact of shocking different nodes.

    We apply a unit shock (100 % distress) to every node one at a time
    and record the resulting system DebtRank.  This reveals which
    institutions pose the greatest systemic risk.
    """
    separator_line("3. Per-Node Systemic Impact (Sensitivity)")

    model = DebtRankModel(variant=DebtRankVariant.LINEAR)

    # Run full sensitivity analysis
    sensitivity = model.sensitivity_analysis(
        G,
        shock_level=1.0,
        max_rounds=50,
    )

    print(f"  System vulnerability  : {sensitivity.system_vulnerability:.6f}")
    print(f"  Concentration risk    : {sensitivity.concentration_risk:.6f}")
    print(f"\n  Top-10 systemically important nodes:")
    for rank, (node, impact) in enumerate(sensitivity.top_k_nodes[:10], 1):
        print(f"    {rank:2d}. Node {node:3d} — impact = {impact:.6f}")

    # Visualize as bar chart
    print(f"\n  Node impact distribution:")
    bar_chart(sensitivity.node_impacts)

    return sensitivity.node_impacts


# =========================================================================
# 4. Contagion path analysis
# =========================================================================

def analyze_contagion_paths(G: nx.DiGraph, dr_result) -> None:
    """Inspect the contagion paths that were activated during propagation.

    Contagion paths reveal the causal chain through which distress
    propagates from the initially shocked node to the rest of the network.
    """
    separator_line("4. Contagion Path Analysis")

    # Show round-by-round propagation history
    if dr_result.round_history:
        print(f"  Propagation history ({dr_result.rounds_propagated} rounds):")
        for r, round_distress in enumerate(dr_result.round_history[:8]):
            active = sum(1 for v in round_distress.values() if v > 0.01)
            max_d = max(round_distress.values()) if round_distress else 0
            print(f"    Round {r:3d}: {active:3d} active nodes, "
                  f"max distress = {max_d:.4f}")
        if dr_result.rounds_propagated > 8:
            print(f"    ... ({dr_result.rounds_propagated - 8} more rounds)")

    # Show final distress distribution
    distress_bins = [0] * 5  # [0-0.2, 0.2-0.4, ..., 0.8-1.0]
    for node_id, level in dr_result.final_distress.items():
        bin_idx = min(int(level * 5), 4)
        distress_bins[bin_idx] += 1

    print(f"\n  Final distress distribution:")
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    for label, count in zip(labels, distress_bins):
        bar = "█" * count
        print(f"    [{label}]: {bar} ({count})")

    # Show most distressed nodes
    sorted_distress = sorted(dr_result.final_distress.items(),
                             key=lambda x: -x[1])
    print(f"\n  Top-5 most distressed nodes:")
    for node, level in sorted_distress[:5]:
        print(f"    Node {node:3d}: distress = {level:.4f}")


# =========================================================================
# 5. Compare with cascade model
# =========================================================================

def compare_with_cascade(G: nx.DiGraph) -> None:
    """Run a threshold cascade model and compare with DebtRank.

    The cascade model uses a binary default/no-default state, while
    DebtRank tracks continuous distress levels.  We compare their
    predictions on the same initial shock.
    """
    separator_line("5. DebtRank vs. Cascade Model Comparison")

    # Pick hub node for consistent comparison
    degrees = dict(G.degree())
    hub_node = max(degrees, key=lambda n: degrees[n])

    # ─── DebtRank ─────────────────────────────────────────────────────
    dr_model = DebtRankModel(variant=DebtRankVariant.LINEAR)
    dr_result = dr_model.compute(
        G,
        initial_shocks={hub_node: 1.0},
        max_rounds=100,
        track_history=True,
    )

    # ─── Cascade model ───────────────────────────────────────────────
    cascade_model = CascadeModel(recovery_rate=0.4)
    cascade_result = cascade_model.simulate_cascade(
        G,
        initial_defaults={hub_node},
        max_rounds=200,
    )

    # ─── Comparison table ─────────────────────────────────────────────
    print(f"  {'Metric':<25s}  {'DebtRank':>12s}  {'Cascade':>12s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}")

    dr_affected = sum(1 for v in dr_result.final_distress.values() if v > 0.01)
    print(f"  {'Affected nodes':<25s}  {dr_affected:>12d}  "
          f"{cascade_result.cascade_size:>12d}")
    print(f"  {'Propagation rounds':<25s}  "
          f"{dr_result.rounds_propagated:>12d}  "
          f"{cascade_result.cascade_rounds:>12d}")
    print(f"  {'System loss (frac)':<25s}  "
          f"{dr_result.loss_fraction:>12.4f}  "
          f"{cascade_result.system_loss:>12.4f}")

    # ─── Cascade path analysis ────────────────────────────────────────
    path_analysis = cascade_model.analyze_cascade_paths(G, cascade_result)
    print(f"\n  Cascade path statistics:")
    print(f"    Total paths        : {len(path_analysis.paths)}")
    print(f"    Mean path length   : {path_analysis.mean_path_length:.2f}")
    print(f"    Max path length    : {path_analysis.max_path_length}")
    print(f"    Branching factor   : {path_analysis.branching_factor:.2f}")
    print(f"    Critical edges     : {len(path_analysis.critical_edges)}")


# =========================================================================
# 6. Tipping-point detection
# =========================================================================

def detect_tipping_points(G: nx.DiGraph) -> None:
    """Find the shock fraction at which a cascade becomes systemic.

    The cascade model searches for a "tipping point" where a small
    increase in the initial shock fraction leads to a discontinuous
    jump in the cascade size.
    """
    separator_line("6. Tipping-Point Detection")

    cascade_model = CascadeModel(recovery_rate=0.4)
    shock_fractions = np.linspace(0.02, 0.30, 15)

    tipping_points = cascade_model.find_tipping_points(
        G,
        n_samples=30,
        shock_fractions=shock_fractions,
        seed=SEED,
    )

    print(f"  {'Shock %':>8s}  {'Cascade %':>10s}  {'Tipping?':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}")

    for tp in tipping_points:
        marker = " ◄── TIPPING" if tp.is_tipping else ""
        print(f"  {tp.shock_size:8.2%}  "
              f"{tp.cascade_fraction:10.4f}  "
              f"{'YES' if tp.is_tipping else 'no':>10s}{marker}")

    # Find the critical tipping point (if any)
    critical = [tp for tp in tipping_points if tp.is_tipping]
    if critical:
        tp = critical[0]
        print(f"\n  ▸ Critical tipping point at {tp.shock_size:.2%} shock")
        print(f"    Cascade fraction jumps to {tp.cascade_fraction:.4f}")
        if tp.critical_nodes:
            print(f"    Critical nodes: {tp.critical_nodes[:5]}")
    else:
        print(f"\n  ▸ No sharp tipping point detected in [{shock_fractions[0]:.0%}, "
              f"{shock_fractions[-1]:.0%}] range.")


# =========================================================================
# 7. DebtRank variant comparison
# =========================================================================

def compare_variants(G: nx.DiGraph) -> None:
    """Compare LINEAR, THRESHOLD, and NONLINEAR DebtRank variants.

    Different propagation functions capture different empirical aspects
    of financial contagion.  The nonlinear variant with exponent > 1
    amplifies large distress signals while damping small ones.
    """
    separator_line("7. DebtRank Variant Comparison")

    hub_node = max(dict(G.degree()), key=lambda n: dict(G.degree())[n])
    initial_shocks = {hub_node: 0.5}

    variants = [
        ("LINEAR", DebtRankVariant.LINEAR, {}),
        ("THRESHOLD", DebtRankVariant.THRESHOLD, {"default_threshold": 0.8}),
        ("NONLINEAR (γ=2)", DebtRankVariant.NONLINEAR, {"nonlinear_exponent": 2.0}),
    ]

    print(f"  {'Variant':<20s}  {'SysRisk':>10s}  {'Cascade':>10s}  "
          f"{'Rounds':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")

    for name, variant, kwargs in variants:
        model = DebtRankModel(variant=variant, **kwargs)
        result = model.compute(G, initial_shocks, max_rounds=100)
        print(f"  {name:<20s}  {result.system_debtrank:>10.6f}  "
              f"{result.cascade_size:>10d}  "
              f"{result.rounds_propagated:>8d}")


# =========================================================================
# Main entry point
# =========================================================================

def main() -> None:
    """Run all DebtRank analysis experiments."""
    print("CausalBound — DebtRank Analysis Example")
    print("=" * 72)

    t0 = time.perf_counter()

    # Generate the network
    G = generate_network(n_nodes=30)

    # Run DebtRank with different shock levels
    sweep_results = run_debtrank_sweep(G)

    # Sensitivity analysis across all nodes
    impacts = run_multinode_shocks(G)

    # Analyze contagion paths for the largest shock
    worst_result = sweep_results[-1]["result"]  # shock = 1.0
    analyze_contagion_paths(G, worst_result)

    # Compare with cascade model
    compare_with_cascade(G)

    # Tipping-point detection
    detect_tipping_points(G)

    # DebtRank variant comparison
    compare_variants(G)

    elapsed = time.perf_counter() - t0
    separator_line("Summary")
    print(f"  Network          : 30-node scale-free")
    print(f"  Shock scenarios  : {len(sweep_results)} severity levels")
    print(f"  Most impactful   : Node {max(impacts, key=lambda k: impacts[k])}")
    print(f"  Total elapsed    : {elapsed:.2f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
