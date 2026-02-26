#!/usr/bin/env python
"""
Causal discovery demo for Causal-Shielded Adaptive Trading.

Demonstrates:
  1. Generate data from a known Structural Causal Model (SCM)
  2. Run the PC algorithm with HSIC kernel independence tests
  3. Compare discovered DAG vs true DAG (SHD, edge F1)
  4. Classify edges as invariant vs regime-specific using SCIT
  5. Compute comprehensive accuracy metrics
  6. Explore Markov equivalence classes
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from causal_trading.market import SyntheticMarketGenerator
from causal_trading.causal import (
    PCAlgorithm,
    HSIC,
    StructuralCausalModel,
    CPDAG,
)
from causal_trading.invariance import SCITAlgorithm, EValueConstructor
from causal_trading.evaluation import CausalAccuracyEvaluator

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
N_FEATURES = 8
N_REGIMES = 3
T = 2000
ALPHA_PC = 0.05


def print_header(title: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")


def adj_to_edge_set(adj: np.ndarray) -> Set[Tuple[int, int]]:
    """Convert adjacency matrix to a set of (parent, child) edges."""
    edges = set()
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j]:
                edges.add((i, j))
    return edges


def print_dag_ascii(adj: np.ndarray, node_names: List[str], title: str) -> None:
    """Print DAG as an adjacency list with ASCII formatting."""
    n = adj.shape[0]
    print(f"\n  {title}")
    print("  " + "─" * 50)
    for i in range(n):
        children = [node_names[j] for j in range(n) if adj[i, j]]
        parents = [node_names[j] for j in range(n) if adj[j, i]]
        if children or parents:
            pa_str = ", ".join(parents) if parents else "∅"
            ch_str = ", ".join(children) if children else "∅"
            print(f"  {node_names[i]:>4s}: Pa={{{pa_str}}} → Ch={{{ch_str}}}")


def print_comparison_matrix(
    true_adj: np.ndarray,
    est_adj: np.ndarray,
    node_names: List[str],
    title: str,
) -> None:
    """Print a side-by-side comparison of true vs estimated DAG."""
    n = true_adj.shape[0]
    print(f"\n  {title}")
    print("  " + "─" * 60)
    print(f"  {'Edge':<15s} {'True':>6s} {'Est.':>6s} {'Status':>10s}")
    print("  " + "─" * 60)

    true_edges = adj_to_edge_set(true_adj)
    est_edges = adj_to_edge_set(est_adj)
    all_edges = true_edges | est_edges

    tp, fp, fn, rev = 0, 0, 0, 0
    for i, j in sorted(all_edges):
        in_true = (i, j) in true_edges
        in_est = (i, j) in est_edges
        reversed_in_est = (j, i) in est_edges

        if in_true and in_est:
            status = "✓ correct"
            tp += 1
        elif in_true and not in_est:
            if reversed_in_est:
                status = "↺ reversed"
                rev += 1
            else:
                status = "✗ missing"
                fn += 1
        else:
            status = "✗ extra"
            fp += 1

        print(f"  {node_names[i]}→{node_names[j]:<10s} "
              f"{'yes':>6s if in_true else 'no':>6s} "
              f"{'yes':>6s if in_est else 'no':>6s} "
              f"{status:>10s}")

    print("  " + "─" * 60)
    print(f"  TP={tp}, FP={fp}, FN={fn}, Reversed={rev}")


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Generate Data from Known SCM
# ═══════════════════════════════════════════════════════════════════════════

def generate_scm_data():
    """Generate regime-switching data with known causal structure."""
    print_header("Part 1: Generate Data from Known SCM")

    gen = SyntheticMarketGenerator(
        n_features=N_FEATURES,
        n_regimes=N_REGIMES,
        invariant_ratio=0.4,
        edge_density=0.15,
        snr=2.0,
        regime_persistence=0.97,
        seed=SEED,
    )
    dataset = gen.generate(T=T)
    gt = dataset.ground_truth
    node_names = [f"X{i}" for i in range(N_FEATURES)]

    print(f"  Generated T={T} observations, p={N_FEATURES} features")
    print(f"  True regimes: K={gt.n_regimes}")

    # Show true DAGs for each regime
    for k in range(N_REGIMES):
        adj = gt.adjacency_matrices[k]
        n_edges = int(adj.sum())
        print(f"\n  Regime {k}: {n_edges} directed edges")
        print_dag_ascii(adj, node_names, f"True DAG for Regime {k}")

    # Show invariant edges
    inv_edges = adj_to_edge_set(gt.invariant_edges)
    print_section("Invariant Edges (present in all regimes)")
    if inv_edges:
        for i, j in sorted(inv_edges):
            print(f"    {node_names[i]} → {node_names[j]}")
    else:
        print("    (none)")
    print(f"  Total invariant edges: {len(inv_edges)}")

    # Show regime-specific edges
    print_section("Regime-Specific Edges")
    for k in range(N_REGIMES):
        regime_edges = adj_to_edge_set(gt.adjacency_matrices[k])
        specific = regime_edges - inv_edges
        if specific:
            print(f"  Regime {k} only:")
            for i, j in sorted(specific):
                print(f"    {node_names[i]} → {node_names[j]}")

    return dataset, node_names


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: PC Algorithm with HSIC
# ═══════════════════════════════════════════════════════════════════════════

def run_pc_algorithm(dataset, node_names):
    """Run the PC algorithm with HSIC independence tests."""
    print_header("Part 2: PC Algorithm with HSIC Independence Tests")

    gt = dataset.ground_truth

    # Run PC on the full dataset (pooled across regimes)
    print_section("PC Algorithm on Pooled Data")
    pc = PCAlgorithm(
        alpha=ALPHA_PC,
        max_cond_size=3,
        stable=True,
    )

    t0 = time.time()
    pc.fit(data=dataset.features, variable_names=node_names)
    elapsed = time.time() - t0
    print(f"  PC algorithm completed in {elapsed:.1f}s")

    pooled_dag = pc.get_dag()
    pooled_cpdag = pc.get_cpdag()
    skeleton = pc.get_skeleton()

    # Convert to adjacency matrix
    pooled_adj = np.zeros((N_FEATURES, N_FEATURES), dtype=bool)
    for i in range(N_FEATURES):
        for j in range(N_FEATURES):
            name_i = node_names[i]
            name_j = node_names[j]
            if pooled_dag is not None and hasattr(pooled_dag, 'has_edge'):
                if pooled_dag.has_edge(name_i, name_j):
                    pooled_adj[i, j] = True
            elif isinstance(pooled_dag, np.ndarray):
                pooled_adj = pooled_dag.astype(bool)
                break
        else:
            continue
        break

    n_edges_pooled = int(pooled_adj.sum())
    print(f"  Pooled DAG: {n_edges_pooled} directed edges")
    print_dag_ascii(pooled_adj, node_names, "Pooled PC DAG")

    # Run PC per regime
    print_section("Per-Regime PC Algorithm")
    per_regime_dags = {}
    for k in range(N_REGIMES):
        mask = gt.regime_labels == k
        if mask.sum() < 50:
            print(f"  Regime {k}: insufficient data ({mask.sum()} obs), skipping")
            continue

        data_k = dataset.features[mask]
        pc_k = PCAlgorithm(alpha=ALPHA_PC, max_cond_size=3, stable=True)
        pc_k.fit(data=data_k, variable_names=node_names)

        dag_k = pc_k.get_dag()
        adj_k = np.zeros((N_FEATURES, N_FEATURES), dtype=bool)
        if dag_k is not None and hasattr(dag_k, 'has_edge'):
            for i in range(N_FEATURES):
                for j in range(N_FEATURES):
                    if dag_k.has_edge(node_names[i], node_names[j]):
                        adj_k[i, j] = True
        elif isinstance(dag_k, np.ndarray):
            adj_k = dag_k.astype(bool)

        per_regime_dags[k] = adj_k
        n_edges_k = int(adj_k.sum())
        print(f"  Regime {k}: {n_edges_k} edges (n={mask.sum()})")

    return pooled_adj, per_regime_dags


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: Compare Discovered vs True DAG
# ═══════════════════════════════════════════════════════════════════════════

def compare_dags(dataset, pooled_adj, per_regime_dags, node_names):
    """Compare discovered DAGs against ground truth."""
    print_header("Part 3: Discovered vs True DAG Comparison")

    gt = dataset.ground_truth
    evaluator = CausalAccuracyEvaluator()

    # Evaluate pooled DAG against each regime's true DAG
    print_section("Pooled DAG vs True DAGs")
    for k in range(N_REGIMES):
        true_adj = gt.adjacency_matrices[k]
        metrics = evaluator.evaluate(true_dag=true_adj, estimated_dag=pooled_adj)

        print(f"\n  Pooled vs Regime {k} Ground Truth:")
        print(f"    SHD:              {metrics.shd}")
        print(f"      Missing:        {metrics.shd_missing}")
        print(f"      Extra:          {metrics.shd_extra}")
        print(f"      Reversed:       {metrics.shd_reversed}")
        print(f"    Edge precision:   {metrics.edge_precision:.4f}")
        print(f"    Edge recall:      {metrics.edge_recall:.4f}")
        print(f"    Edge F1:          {metrics.edge_f1:.4f}")
        print(f"    Adjacency F1:     {metrics.adjacency_f1:.4f}")

    # Evaluate per-regime DAGs
    if per_regime_dags:
        print_section("Per-Regime DAGs vs True DAGs")
        for k in sorted(per_regime_dags.keys()):
            if k not in gt.adjacency_matrices:
                continue
            true_adj = gt.adjacency_matrices[k]
            est_adj = per_regime_dags[k]
            metrics = evaluator.evaluate(true_dag=true_adj, estimated_dag=est_adj)

            print(f"\n  Regime {k} (per-regime PC vs truth):")
            print(f"    SHD:              {metrics.shd}")
            print(f"    Edge precision:   {metrics.edge_precision:.4f}")
            print(f"    Edge recall:      {metrics.edge_recall:.4f}")
            print(f"    Edge F1:          {metrics.edge_f1:.4f}")

            print_comparison_matrix(
                true_adj, est_adj, node_names,
                f"Regime {k}: True vs Estimated"
            )

    # Comparison summary table
    print_section("Summary: SHD Across Methods")
    print(f"  {'Method':<25s} {'Regime 0':>10s} {'Regime 1':>10s} {'Regime 2':>10s}")
    print("  " + "─" * 58)

    # Pooled
    shds = []
    for k in range(N_REGIMES):
        m = evaluator.evaluate(true_dag=gt.adjacency_matrices[k], estimated_dag=pooled_adj)
        shds.append(m.shd)
    print(f"  {'Pooled PC':<25s} {shds[0]:>10d} {shds[1]:>10d} {shds[2]:>10d}")

    # Per-regime
    if per_regime_dags:
        shds2 = []
        for k in range(N_REGIMES):
            if k in per_regime_dags and k in gt.adjacency_matrices:
                m = evaluator.evaluate(
                    true_dag=gt.adjacency_matrices[k],
                    estimated_dag=per_regime_dags[k],
                )
                shds2.append(m.shd)
            else:
                shds2.append(-1)
        vals = [f"{s:>10d}" if s >= 0 else f"{'N/A':>10s}" for s in shds2]
        print(f"  {'Per-regime PC':<25s} {vals[0]} {vals[1]} {vals[2]}")


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Invariance Testing with SCIT
# ═══════════════════════════════════════════════════════════════════════════

def run_invariance_testing(dataset, per_regime_dags, node_names):
    """Classify edges as invariant or regime-specific using SCIT."""
    print_header("Part 4: Invariance Testing (SCIT with e-values)")

    gt = dataset.ground_truth

    # Build union DAG from per-regime discoveries
    union_dag: Dict[str, List[str]] = {name: [] for name in node_names}

    # Use ground truth DAGs for a cleaner demo if per-regime results are sparse
    source_dags = per_regime_dags if per_regime_dags else gt.adjacency_matrices
    for k, adj in source_dags.items():
        if isinstance(adj, np.ndarray):
            for i in range(N_FEATURES):
                for j in range(N_FEATURES):
                    if adj[i, j]:
                        parent = node_names[i]
                        child = node_names[j]
                        if child not in union_dag[parent]:
                            union_dag[parent].append(child)

    # Count total edges in union
    total_union = sum(len(children) for children in union_dag.values())
    print(f"  Union DAG: {total_union} directed edges")

    # Run SCIT
    scit = SCITAlgorithm(
        alpha=0.05,
        min_samples_per_regime=10,
        doubly_robust=True,
        early_stop=True,
    )

    t0 = time.time()
    result = scit.fit(
        data=dataset.features,
        regimes=gt.regime_labels,  # Use ground truth for cleaner demo
        dag=union_dag,
        node_names=node_names,
    )
    elapsed = time.time() - t0
    print(f"  SCIT completed in {elapsed:.1f}s")

    # Classify edges
    classifications = scit.classify_edges(union_dag)
    n_invariant = sum(1 for c in classifications.values() if c.is_invariant)
    n_regime_spec = sum(1 for c in classifications.values() if not c.is_invariant)

    print(f"\n  Classification results:")
    print(f"    Invariant edges:       {n_invariant}")
    print(f"    Regime-specific edges: {n_regime_spec}")
    print(f"    Total tested:          {len(classifications)}")

    # Detailed classification
    print_section("Edge Classification Details")
    print(f"  {'Edge':<15s} {'Classification':>16s} {'e-value':>10s}")
    print("  " + "─" * 45)

    for (u, v), cls in sorted(classifications.items()):
        status = "INVARIANT" if cls.is_invariant else "REGIME-SPECIFIC"
        ev = getattr(cls, 'e_value', None)
        ev_str = f"{ev:.4f}" if ev is not None else "N/A"
        print(f"  {u}→{v:<10s} {status:>16s} {ev_str:>10s}")

    # Compare with ground truth invariant edges
    print_section("Invariance Classification Accuracy")
    true_inv = adj_to_edge_set(gt.invariant_edges)
    est_inv = set()
    est_spec = set()

    for (u, v), cls in classifications.items():
        ui = int(u.replace("X", "")) if isinstance(u, str) else u
        vi = int(v.replace("X", "")) if isinstance(v, str) else v
        if cls.is_invariant:
            est_inv.add((ui, vi))
        else:
            est_spec.add((ui, vi))

    # Precision / recall for invariant edge detection
    tp_inv = len(est_inv & true_inv)
    fp_inv = len(est_inv - true_inv)
    fn_inv = len(true_inv - est_inv)

    prec_inv = tp_inv / (tp_inv + fp_inv) if (tp_inv + fp_inv) > 0 else 0
    rec_inv = tp_inv / (tp_inv + fn_inv) if (tp_inv + fn_inv) > 0 else 0
    f1_inv = 2 * prec_inv * rec_inv / (prec_inv + rec_inv) if (prec_inv + rec_inv) > 0 else 0

    print(f"  True invariant edges:     {len(true_inv)}")
    print(f"  Estimated invariant:      {len(est_inv)}")
    print(f"  True positives:           {tp_inv}")
    print(f"  False positives:          {fp_inv}")
    print(f"  False negatives:          {fn_inv}")
    print(f"  Invariance precision:     {prec_inv:.4f}")
    print(f"  Invariance recall:        {rec_inv:.4f}")
    print(f"  Invariance F1:            {f1_inv:.4f}")

    return classifications


# ═══════════════════════════════════════════════════════════════════════════
# Part 5: HSIC Independence Test Deep Dive
# ═══════════════════════════════════════════════════════════════════════════

def hsic_deep_dive(dataset):
    """Demonstrate HSIC kernel independence testing in detail."""
    print_header("Part 5: HSIC Independence Test Deep Dive")

    gt = dataset.ground_truth
    rng = np.random.default_rng(SEED)

    hsic = HSIC(
        unbiased=True,
        n_permutations=500,
        alpha=0.05,
    )

    # Test independence between pairs of features
    print_section("Pairwise HSIC Tests (pooled data)")
    print(f"  {'Pair':<12s} {'HSIC':>10s} {'p-value':>10s} {'Dependent':>10s} {'True Edge':>10s}")
    print("  " + "─" * 55)

    n_test_pairs = min(15, N_FEATURES * (N_FEATURES - 1) // 2)
    pairs_tested = 0
    correct = 0
    total = 0

    for i in range(N_FEATURES):
        for j in range(i + 1, N_FEATURES):
            if pairs_tested >= n_test_pairs:
                break
            X = dataset.features[:, i:i+1]
            Y = dataset.features[:, j:j+1]

            result = hsic.test(X, Y, seed=SEED + pairs_tested)
            is_dep = result.p_value < 0.05

            # Check ground truth: is there any edge between i and j in any regime?
            has_true_edge = False
            for k in range(N_REGIMES):
                if gt.adjacency_matrices[k][i, j] or gt.adjacency_matrices[k][j, i]:
                    has_true_edge = True
                    break

            status = "✓" if is_dep == has_true_edge else "✗"
            if is_dep == has_true_edge:
                correct += 1
            total += 1

            print(f"  X{i}↔X{j:<5s} {result.statistic:>10.4f} {result.p_value:>10.4f} "
                  f"{'yes':>10s if is_dep else 'no':>10s} "
                  f"{'yes':>10s if has_true_edge else 'no':>10s} {status}")
            pairs_tested += 1
        if pairs_tested >= n_test_pairs:
            break

    if total > 0:
        print(f"\n  Concordance with ground truth: {correct}/{total} ({100*correct/total:.1f}%)")

    # Per-regime HSIC comparison
    print_section("Per-Regime HSIC for a Selected Pair")
    test_i, test_j = 0, 1
    print(f"  Testing X{test_i} vs X{test_j} in each regime:\n")

    for k in range(N_REGIMES):
        mask = gt.regime_labels == k
        if mask.sum() < 30:
            print(f"  Regime {k}: insufficient data ({mask.sum()} obs)")
            continue

        X_k = dataset.features[mask, test_i:test_i+1]
        Y_k = dataset.features[mask, test_j:test_j+1]
        result_k = hsic.test(X_k, Y_k, seed=SEED + k)

        has_edge = (gt.adjacency_matrices[k][test_i, test_j] or
                    gt.adjacency_matrices[k][test_j, test_i])

        print(f"  Regime {k} (n={mask.sum()}):")
        print(f"    HSIC statistic: {result_k.statistic:.6f}")
        print(f"    p-value:        {result_k.p_value:.6f}")
        print(f"    Dependent:      {'yes' if result_k.p_value < 0.05 else 'no'}")
        print(f"    True edge:      {'yes' if has_edge else 'no'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run all causal discovery demonstrations."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Causal-Shielded Adaptive Trading — Causal Discovery Demo ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Step 1: Generate data
    dataset, node_names = generate_scm_data()

    # Step 2: PC algorithm
    pooled_adj, per_regime_dags = run_pc_algorithm(dataset, node_names)

    # Step 3: Compare with ground truth
    compare_dags(dataset, pooled_adj, per_regime_dags, node_names)

    # Step 4: Invariance testing
    run_invariance_testing(dataset, per_regime_dags, node_names)

    # Step 5: HSIC deep dive
    hsic_deep_dive(dataset)

    print("\n" + "═" * 72)
    print("  Causal discovery demo complete.")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
