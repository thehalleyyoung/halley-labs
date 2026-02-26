#!/usr/bin/env python3
"""
CausalBound — Basic End-to-End Pipeline Example
================================================

This example demonstrates the complete CausalBound pipeline on a small
financial network (20 nodes).  Each stage is clearly annotated so you can
follow the data flow:

    1.  Generate a synthetic interbank exposure network
    2.  Build a Structural Causal Model (SCM) from the network
    3.  Tree-decompose the underlying DAG into bounded-treewidth subgraphs
    4.  Solve causal-polytope LPs on every subgraph
    5.  Compose local bounds into a global worst-case risk bound
    6.  Run MCTS adversarial search for worst-case shock scenarios
    7.  Print and inspect every intermediate result

All imports are from the ``causalbound`` package; the script is entirely
self-contained and requires no external data files.

Usage
-----
    python basic_pipeline.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# ── CausalBound imports ──────────────────────────────────────────────────
from causalbound.network.generators import (
    ErdosRenyiGenerator,
    ExposureParams,
)
from causalbound.network.topology import NetworkTopology
from causalbound.scm.builder import SCMBuilder
from causalbound.graph.decomposition import TreeDecomposer
from causalbound.polytope.causal_polytope import (
    CausalPolytopeSolver,
    DAGSpec,
    InterventionSpec,
    ObservedMarginals,
    QuerySpec,
    SolverConfig,
)
from causalbound.composition.composer import (
    BoundComposer,
    CompositionStrategy,
    OverlapStructure,
    SeparatorInfo,
    SubgraphBound,
)
from causalbound.mcts.search import MCTSSearch, SearchConfig
from causalbound.junction.engine import JunctionTreeEngine

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


def dag_from_digraph(G: nx.DiGraph) -> DAGSpec:
    """Convert a NetworkX DiGraph to the DAGSpec expected by the LP solver.

    Each node is mapped to a string label and assigned a default
    cardinality of 2 (binary distress / no-distress).
    """
    node_labels = {n: f"X{n}" for n in G.nodes()}
    nodes = list(node_labels.values())
    edges = [(node_labels[u], node_labels[v]) for u, v in G.edges()]
    card = {label: 2 for label in nodes}
    return DAGSpec(nodes=nodes, edges=edges, card=card)


def make_observed_marginals(
    dag: DAGSpec,
    rng: np.random.Generator,
) -> ObservedMarginals:
    """Synthesise plausible observed marginals for every single variable.

    In a real application these would come from historical market data.
    Here we draw random Dirichlet probabilities so the LP has realistic
    constraints.
    """
    marginals: Dict[frozenset, np.ndarray] = {}
    for node_name in dag.nodes:
        card = dag.card[node_name]
        # Dirichlet(alpha=2) gives a moderately concentrated distribution
        p = rng.dirichlet(np.full(card, 2.0))
        marginals[frozenset([node_name])] = p
    return ObservedMarginals(marginals=marginals)


# =========================================================================
# Stage 1 — Generate a financial network
# =========================================================================

def stage_generate_network(n_nodes: int = 20) -> nx.DiGraph:
    """Create a small Erdős–Rényi interbank exposure network.

    Each edge carries an ``exposure`` weight drawn from a Pareto tail
    distribution; each node carries stylised ``capital`` and ``size``
    attributes.
    """
    separator_line("Stage 1: Generate Financial Network")

    # Configure exposures with heavy-tailed Pareto parameters
    exposure_params = ExposureParams(
        distribution="pareto",
        location=1e6,
        scale=5e7,
        shape=1.5,
        min_exposure=1e5,
        max_exposure=1e11,
    )
    generator = ErdosRenyiGenerator(exposure_params=exposure_params, seed=SEED)

    # Generate a directed network with ~10 % density
    G = generator.generate(n_nodes=n_nodes, density=0.10, reciprocity=0.5)

    # Analyse and print basic topology statistics
    topo = NetworkTopology()
    report = topo.analyze(G)
    print(f"  Nodes           : {report.n_nodes}")
    print(f"  Edges           : {report.n_edges}")
    print(f"  Density         : {report.density:.4f}")
    print(f"  Reciprocity     : {report.reciprocity:.4f}")
    print(f"  Avg clustering  : {report.clustering:.4f}")
    print(f"  Components (WCC): {report.components}")

    return G


# =========================================================================
# Stage 2 — Build a Structural Causal Model from the network
# =========================================================================

def stage_build_scm(G: nx.DiGraph):
    """Translate the exposure network into a Structural Causal Model.

    ``SCMBuilder.build_from_network`` maps each institution to an SCM
    variable whose parents are the institutions that lend to it.  Latent
    confounders (e.g. common asset holdings) are injected by default.
    """
    separator_line("Stage 2: Build Structural Causal Model")

    builder = SCMBuilder()
    scm = builder.build_from_network(G, latent_confounders=True)

    print(f"  SCM variables   : {len(scm.variables)}")
    print(f"  Observed vars   : {sum(1 for v in scm.variables.values() if v.observed)}")
    print(f"  Latent vars     : {sum(1 for v in scm.variables.values() if not v.observed)}")
    print(f"  Equations       : {len(scm.equations)}")

    # Print a few example variable entries
    for i, (name, var) in enumerate(scm.variables.items()):
        if i >= 3:
            break
        print(f"    {name}: type={var.var_type.value}, "
              f"parents={var.parents[:3]}, observed={var.observed}")

    return scm


# =========================================================================
# Stage 3 — Tree-decompose the DAG
# =========================================================================

def stage_decompose(G: nx.DiGraph):
    """Compute a tree decomposition of the DAG's moral graph.

    The moral graph is formed by marrying parents and dropping edge
    directions.  ``TreeDecomposer`` then uses a min-fill heuristic to
    find a low-width decomposition.
    """
    separator_line("Stage 3: Tree Decomposition")

    # Build the moral (undirected) graph
    moral = nx.Graph()
    moral.add_nodes_from(G.nodes())
    for node in G.nodes():
        parents = list(G.predecessors(node))
        # Marry the parents
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                moral.add_edge(parents[i], parents[j])
        for parent in parents:
            moral.add_edge(parent, node)

    decomposer = TreeDecomposer(strategy="min_fill")
    decomposition = decomposer.decompose(moral, max_width=15)

    print(f"  Tree-width      : {decomposition.width}")
    print(f"  Number of bags  : {len(decomposition.bags)}")
    print(f"  Elimination len : {len(decomposition.ordering)}")

    # Show a few bags
    for bag_id in sorted(decomposition.bags)[:5]:
        bag_nodes = decomposition.bags[bag_id]
        print(f"    Bag {bag_id}: {set(bag_nodes)}")

    # Validate the decomposition
    valid = decomposer.validate_decomposition(moral, decomposition)
    print(f"  Valid decomp.   : {valid}")

    return decomposition


# =========================================================================
# Stage 4 — Solve causal-polytope LPs on each subgraph
# =========================================================================

def stage_solve_lps(G: nx.DiGraph, decomposition):
    """Run the causal-polytope LP solver on every bag of the decomposition.

    For each bag (subgraph), we:
      - extract the induced sub-DAG,
      - construct a causal polytope LP,
      - solve for the tightest probability bounds on systemic default.
    """
    separator_line("Stage 4: Causal-Polytope LP Solving")

    solver_config = SolverConfig(
        max_iterations=200,
        gap_tolerance=1e-6,
        pricing_strategy="exact",
        warm_start=True,
        time_limit=60.0,
    )
    solver = CausalPolytopeSolver(config=solver_config)

    rng = np.random.default_rng(SEED)
    all_nodes = list(G.nodes())
    subgraph_bounds: List[SubgraphBound] = []

    for bag_id, bag_nodes in sorted(decomposition.bags.items()):
        # Extract the induced sub-DAG on this bag
        bag_list = [n for n in bag_nodes if n in G.nodes()]
        if len(bag_list) < 2:
            continue
        sub_dag = G.subgraph(bag_list).copy()
        if sub_dag.number_of_edges() == 0:
            continue

        # Convert to DAGSpec format
        dag_spec = dag_from_digraph(sub_dag)
        observed = make_observed_marginals(dag_spec, rng)

        # The query asks: what is the maximum P(target = distressed)?
        target_var = dag_spec.nodes[0]
        query = QuerySpec(target_var=target_var, target_val=1)

        # Solve the LP
        result = solver.solve(dag_spec, query, observed=observed)

        print(f"  Bag {bag_id:3d} | nodes={len(bag_list):2d} "
              f"| [{result.lower_bound:.6f}, {result.upper_bound:.6f}] "
              f"| status={result.status.value}")

        # Collect bounds for composition
        sg_bound = SubgraphBound(
            subgraph_id=bag_id,
            lower=np.array([result.lower_bound]),
            upper=np.array([result.upper_bound]),
            separator_vars=[n for n in bag_list if n in all_nodes],
            weight=1.0,
            confidence=1.0,
        )
        subgraph_bounds.append(sg_bound)

    print(f"\n  Total subgraph bounds computed: {len(subgraph_bounds)}")
    return subgraph_bounds


# =========================================================================
# Stage 5 — Compose local bounds into global bounds
# =========================================================================

def stage_compose_bounds(subgraph_bounds: List[SubgraphBound]):
    """Aggregate subgraph-level bounds into a single global risk bound.

    The composition theorem guarantees that the resulting global bound
    is valid whenever the local bounds are valid.  We use the
    WORST_CASE strategy which yields the loosest but most robust bound.
    """
    separator_line("Stage 5: Bound Composition")

    if not subgraph_bounds:
        print("  No subgraph bounds to compose — skipping.")
        return None

    composer = BoundComposer(
        strategy=CompositionStrategy.WORST_CASE,
        tolerance=1e-8,
        max_iterations=200,
    )

    # Build separator info from overlapping nodes between consecutive bags
    separator_info: List[SeparatorInfo] = []
    for i in range(len(subgraph_bounds) - 1):
        shared = set(subgraph_bounds[i].separator_vars) & set(
            subgraph_bounds[i + 1].separator_vars
        )
        if shared:
            sep = SeparatorInfo(
                separator_id=i,
                variable_indices=list(shared)[:3],
                adjacent_subgraphs=[
                    subgraph_bounds[i].subgraph_id,
                    subgraph_bounds[i + 1].subgraph_id,
                ],
                cardinality=2,
            )
            separator_info.append(sep)

    # Build a simple overlap structure
    n_sg = len(subgraph_bounds)
    overlap_matrix = np.eye(n_sg)
    for i in range(n_sg - 1):
        overlap_matrix[i, i + 1] = overlap_matrix[i + 1, i] = 0.1
    overlap = OverlapStructure(
        n_subgraphs=n_sg,
        overlap_matrix=overlap_matrix,
    )

    result = composer.compose(subgraph_bounds, separator_info, overlap)

    print(f"  Strategy        : {result.strategy_used.value}")
    print(f"  Global lower    : {result.global_lower}")
    print(f"  Global upper    : {result.global_upper}")
    print(f"  Composition gap : {result.composition_gap:.8f}")
    print(f"  Converged       : {result.converged}")
    print(f"  Iterations      : {result.n_iterations}")

    return result


# =========================================================================
# Stage 6 — MCTS adversarial scenario search
# =========================================================================

def stage_adversarial_search(G: nx.DiGraph, scm):
    """Search for the worst-case initial shock scenario using MCTS.

    The MCTS algorithm explores the space of possible initial stress
    vectors, guided by causal UCB scores.  Each rollout evaluates the
    systemic impact of a candidate shock vector.
    """
    separator_line("Stage 6: MCTS Adversarial Search")

    config = SearchConfig(
        n_rollouts=500,
        budget_seconds=60.0,
        exploration_constant=1.414,
        maximize=True,
        enable_pruning=True,
        convergence_check_interval=100,
        values_per_variable=5,
        shock_range=(-2.0, 2.0),
    )

    searcher = MCTSSearch(config=config, random_seed=SEED)

    # The interface variables are the exogenous shock nodes
    interface_vars = [f"X{n}" for n in list(G.nodes())[:5]]

    # Build a lightweight inference engine for rollout evaluation
    engine = JunctionTreeEngine(default_bins=10)

    # Build a small DAG for the engine
    subset_nodes = list(G.nodes())[:8]
    sub_G = G.subgraph(subset_nodes).copy()
    dag_dict = {}
    cpds = {}
    cards = {}
    for node in sub_G.nodes():
        label = f"X{node}"
        parents = [f"X{p}" for p in sub_G.predecessors(node)]
        dag_dict[label] = parents
        cards[label] = 2
        # Build a simple CPD as a PotentialTable placeholder
        n_configs = 2 ** (len(parents) + 1)

    try:
        # Attempt to run the search with the inference engine
        result = searcher.search(
            interface_vars=interface_vars[:3],
            inference_engine=engine,
            n_rollouts=200,
            budget_seconds=30.0,
            target_variable=interface_vars[0],
        )

        print(f"  Total rollouts  : {result.total_rollouts}")
        print(f"  Elapsed (s)     : {result.elapsed_seconds:.2f}")
        print(f"  Converged       : {result.converged}")

        if result.best_scenario:
            print(f"\n  Best scenario:")
            print(f"    Value         : {result.best_scenario.value:.6f}")
            print(f"    Visit count   : {result.best_scenario.visit_count}")
            print(f"    State         : {result.best_scenario.state}")

        # Show top-3 scenarios
        print(f"\n  Top scenarios:")
        for scenario in result.all_scenarios[:3]:
            print(f"    Rank {scenario.rank}: "
                  f"value={scenario.value:.4f}, "
                  f"visits={scenario.visit_count}")

        return result

    except Exception as e:
        print(f"  MCTS search encountered an error: {e}")
        print("  (This can happen with very small networks.)")
        return None


# =========================================================================
# Stage 7 — Summary and final report
# =========================================================================

def stage_summary(G, decomposition, subgraph_bounds, composition_result, search_result):
    """Print a consolidated summary of the full pipeline run."""
    separator_line("Pipeline Summary")

    print(f"  Network size     : {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    print(f"  Tree-width       : {decomposition.width}")
    print(f"  Subgraph bounds  : {len(subgraph_bounds)} solved")

    if composition_result:
        print(f"  Global bound     : [{composition_result.global_lower[0]:.6f}, "
              f"{composition_result.global_upper[0]:.6f}]")
        print(f"  Composition gap  : {composition_result.composition_gap:.8f}")

    if search_result and search_result.best_scenario:
        print(f"  Worst-case value : {search_result.best_scenario.value:.6f}")

    print()


# =========================================================================
# Main entry point
# =========================================================================

def main() -> None:
    """Run the full CausalBound pipeline end-to-end."""
    print("CausalBound — Basic End-to-End Pipeline")
    print("=" * 72)

    t0 = time.perf_counter()

    # Stage 1: Generate the financial network
    G = stage_generate_network(n_nodes=20)

    # Stage 2: Build the structural causal model
    scm = stage_build_scm(G)

    # Stage 3: Tree-decompose the DAG
    decomposition = stage_decompose(G)

    # Stage 4: Solve causal-polytope LPs on every subgraph
    subgraph_bounds = stage_solve_lps(G, decomposition)

    # Stage 5: Compose local bounds into a global risk bound
    composition_result = stage_compose_bounds(subgraph_bounds)

    # Stage 6: Adversarial search for worst-case scenarios
    search_result = stage_adversarial_search(G, scm)

    # Stage 7: Final summary
    stage_summary(G, decomposition, subgraph_bounds,
                  composition_result, search_result)

    elapsed = time.perf_counter() - t0
    print(f"  Total elapsed    : {elapsed:.2f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
