#!/usr/bin/env python3
"""
CausalBound — Historical Crisis Simulation
===========================================

This example reconstructs the topology of the 2008 Global Financial
Crisis (GFC) and runs the full CausalBound pipeline to discover
worst-case systemic scenarios.  The discovered scenarios are then
compared with the *known* crisis mechanisms that unfolded in 2007–2009.

Pipeline:

    1. Reconstruct 2008 GFC network topology
       (stylised version: 18 major institutions, CDS exposures)
    2. Build an SCM with CDS instrument modelling
    3. Run the full CausalBound pipeline (decompose → LP → compose)
    4. Run MCTS adversarial search
    5. Compare discovered scenarios with documented crisis mechanisms
    6. Report structural similarity metrics

Data sources: The topology and exposure magnitudes are stylised
reconstructions based on public BIS/FSB data and academic literature
(see Battiston et al. 2012, Cont et al. 2013).

Usage
-----
    python crisis_simulation.py
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# ── CausalBound imports ──────────────────────────────────────────────────
from causalbound.evaluation.crisis_reconstruction import (
    CrisisReconstructor,
    CrisisTopology,
    ComparisonResult,
    StructuralSimilarity,
)
from causalbound.evaluation.metrics import MetricsComputer
from causalbound.network.topology import NetworkTopology, CentralityMethod
from causalbound.scm.builder import SCMBuilder
from causalbound.instruments.cds import CDSModel
from causalbound.instruments.discretization import (
    InstrumentDiscretizer,
    DiscretizationStrategy,
)
from causalbound.graph.decomposition import TreeDecomposer
from causalbound.polytope.causal_polytope import (
    CausalPolytopeSolver,
    DAGSpec,
    QuerySpec,
    SolverConfig,
    ObservedMarginals,
)
from causalbound.composition.composer import (
    BoundComposer,
    CompositionStrategy,
    SubgraphBound,
    SeparatorInfo,
    OverlapStructure,
)
from causalbound.mcts.search import MCTSSearch, SearchConfig
from causalbound.junction.engine import JunctionTreeEngine
from causalbound.contagion.debtrank import DebtRankModel, DebtRankVariant

# ── Reproducibility ──────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


# =========================================================================
# Helper utilities
# =========================================================================

def separator_line(title: str, width: int = 72) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


# =========================================================================
# Known crisis mechanisms (ground truth for comparison)
# =========================================================================

# These are the major contagion channels documented in the 2008 GFC
# literature.  Each is a list of institution names (or types) that
# formed the critical path.

KNOWN_CRISIS_MECHANISMS: List[Dict[str, Any]] = [
    {
        "name": "Subprime → MBS → CDS Cascade",
        "description": "Subprime mortgage defaults triggered MBS write-downs "
                       "which activated CDS protection payments, amplifying "
                       "losses through AIG's massive CDS book.",
        "critical_nodes": ["AIG", "Lehman", "BearStearns"],
        "channel": "credit_derivatives",
    },
    {
        "name": "Repo Funding Freeze",
        "description": "Loss of confidence in counterparty solvency led to "
                       "withdrawal of short-term repo funding, creating "
                       "liquidity spirals at broker-dealers.",
        "critical_nodes": ["Lehman", "BearStearns", "MerrillLynch"],
        "channel": "funding_liquidity",
    },
    {
        "name": "Fire-Sale Contagion",
        "description": "Forced deleveraging by distressed institutions "
                       "depressed asset prices, triggering mark-to-market "
                       "losses at other holders of the same assets.",
        "critical_nodes": ["Lehman", "MorganStanley", "GoldmanSachs"],
        "channel": "fire_sale",
    },
    {
        "name": "Interbank Exposure Concentration",
        "description": "Heavy counterparty concentration in the interbank "
                       "market meant failure of one G-SIB imposed large "
                       "direct losses on a handful of others.",
        "critical_nodes": ["JPMorgan", "Citigroup", "BankOfAmerica"],
        "channel": "direct_exposure",
    },
]


# =========================================================================
# 1. Reconstruct 2008 GFC network topology
# =========================================================================

def reconstruct_gfc_network() -> CrisisTopology:
    """Use the CrisisReconstructor to build a stylised GFC topology.

    The reconstructor provides pre-built topologies for major
    historical crises based on publicly available data.
    """
    separator_line("1. Reconstruct 2008 GFC Network Topology")

    reconstructor = CrisisReconstructor(seed=SEED)
    crisis = reconstructor.reconstruct("gfc_2008")

    G = crisis.graph
    print(f"  Crisis name      : {crisis.name}")
    print(f"  Nodes            : {G.number_of_nodes()}")
    print(f"  Edges            : {G.number_of_edges()}")

    # Analyse topology
    topo = NetworkTopology()
    report = topo.analyze(G)
    print(f"  Density          : {report.density:.4f}")
    print(f"  Reciprocity      : {report.reciprocity:.4f}")
    print(f"  Avg clustering   : {report.clustering:.4f}")

    # Show institutions
    if crisis.node_metadata:
        print(f"\n  Institutions:")
        for i, (name, meta) in enumerate(
            sorted(crisis.node_metadata.items())[:12]
        ):
            inst_type = meta.get("type", "unknown")
            size = meta.get("size", 0)
            print(f"    {i+1:2d}. {name:<20s} type={inst_type:<12s} "
                  f"size=${size/1e9:.0f}B")
        if len(crisis.node_metadata) > 12:
            print(f"    ... and {len(crisis.node_metadata) - 12} more")

    # Show known historical scenarios
    if crisis.known_scenarios:
        print(f"\n  Known historical scenarios: {len(crisis.known_scenarios)}")
        for sc in crisis.known_scenarios[:3]:
            print(f"    • {sc.get('name', 'unnamed')}")

    # Show historical losses
    if crisis.historical_losses:
        print(f"\n  Historical loss estimates:")
        for entity, loss in sorted(
            crisis.historical_losses.items(),
            key=lambda kv: -kv[1],
        )[:8]:
            print(f"    {entity:<20s}: ${loss/1e9:.1f}B")

    return crisis


# =========================================================================
# 2. Build SCM with CDS instruments
# =========================================================================

def build_scm_with_cds(crisis: CrisisTopology):
    """Construct an SCM from the crisis topology.

    CDS instruments are attached to each edge to model the credit
    derivative exposures that were central to the 2008 crisis.
    """
    separator_line("2. Build SCM with CDS Instruments")

    G = crisis.graph

    # ─── Build CDS models for major exposures ────────────────────────
    print("  Building CDS models for edge exposures...")
    cds_models: Dict[Tuple, CDSModel] = {}

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1e7)
        # Model each edge as a CDS contract
        cds = CDSModel(
            notional=float(weight),
            spread=100.0 + np.random.uniform(-50, 150),
            tenor=5.0,
            recovery=0.4,
            frequency=0.25,
            reference_entity=str(v),
        )
        cds_models[(u, v)] = cds

    print(f"  CDS models built : {len(cds_models)}")

    # ─── Discretize CDS payoffs ──────────────────────────────────────
    discretizer = InstrumentDiscretizer(
        default_n_bins=20,
        default_strategy=DiscretizationStrategy.CDS_FOCUSED,
    )

    n_discretized = 0
    for (u, v), cds in list(cds_models.items())[:10]:
        try:
            disc_result = discretizer.discretize_payoff(
                payoff_fn=lambda t, _cds=cds: np.array([
                    _cds.payoff(float(ti)) for ti in t
                ]),
                domain=(0.0, 5.0),
                n_bins=15,
            )
            n_discretized += 1
        except Exception:
            pass

    print(f"  Payoffs discret. : {n_discretized}")

    # ─── Build the SCM ───────────────────────────────────────────────
    builder = SCMBuilder()
    scm = builder.build_from_network(G, latent_confounders=True)

    print(f"\n  SCM statistics:")
    print(f"    Variables      : {len(scm.variables)}")
    print(f"    Observed       : {sum(1 for v in scm.variables.values() if v.observed)}")
    print(f"    Latent         : {sum(1 for v in scm.variables.values() if not v.observed)}")
    print(f"    Equations      : {len(scm.equations)}")

    return scm, cds_models


# =========================================================================
# 3. Run full CausalBound pipeline
# =========================================================================

def run_causalbound_pipeline(
    crisis: CrisisTopology,
    scm,
) -> Tuple[List[SubgraphBound], Optional[Any]]:
    """Execute the decompose → LP → compose pipeline.

    This is the core CausalBound computation: we decompose the DAG,
    solve bounded-treewidth causal polytope LPs on each piece, then
    compose the local bounds into a valid global bound.
    """
    separator_line("3. Run Full CausalBound Pipeline")

    G = crisis.graph

    # ── 3a. Tree decomposition ────────────────────────────────────────
    print("  Step 3a: Tree decomposition...")

    moral = nx.Graph()
    moral.add_nodes_from(G.nodes())
    for node in G.nodes():
        parents = list(G.predecessors(node))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                moral.add_edge(parents[i], parents[j])
        for parent in parents:
            moral.add_edge(parent, node)

    decomposer = TreeDecomposer(strategy="min_fill")
    decomposition = decomposer.decompose(moral, max_width=15)

    print(f"    Tree-width     : {decomposition.width}")
    print(f"    Bags           : {len(decomposition.bags)}")

    # ── 3b. Causal-polytope LP on each subgraph ──────────────────────
    print("  Step 3b: Causal-polytope LP solving...")

    solver = CausalPolytopeSolver(config=SolverConfig(
        max_iterations=150,
        gap_tolerance=1e-6,
        time_limit=30.0,
    ))

    rng = np.random.default_rng(SEED)
    subgraph_bounds: List[SubgraphBound] = []
    total_solve_time = 0.0
    n_solved = 0

    for bag_id, bag_nodes in sorted(decomposition.bags.items()):
        bag_list = [n for n in bag_nodes if n in G.nodes()]
        if len(bag_list) < 2:
            continue

        sub_dag = G.subgraph(bag_list).copy()
        if sub_dag.number_of_edges() == 0:
            continue

        # Build DAGSpec
        node_labels = {n: f"N{n}" for n in sub_dag.nodes()}
        dag_spec = DAGSpec(
            nodes=list(node_labels.values()),
            edges=[(node_labels[u], node_labels[v])
                   for u, v in sub_dag.edges()],
            card={label: 2 for label in node_labels.values()},
        )

        # Observed marginals
        marginals = {}
        for name in dag_spec.nodes:
            marginals[frozenset([name])] = rng.dirichlet([2.0, 2.0])
        observed = ObservedMarginals(marginals=marginals)

        # Query: max P(target = distressed)
        target = dag_spec.nodes[0]
        query = QuerySpec(target_var=target, target_val=1)

        t0 = time.perf_counter()
        result = solver.solve(dag_spec, query, observed=observed)
        solve_time = time.perf_counter() - t0
        total_solve_time += solve_time
        n_solved += 1

        subgraph_bounds.append(SubgraphBound(
            subgraph_id=bag_id,
            lower=np.array([result.lower_bound]),
            upper=np.array([result.upper_bound]),
            separator_vars=[n for n in bag_list],
        ))

    print(f"    Subgraphs solved: {n_solved}")
    print(f"    Total solve time: {total_solve_time:.2f}s")

    # ── 3c. Bound composition ────────────────────────────────────────
    print("  Step 3c: Bound composition...")

    composition_result = None
    if len(subgraph_bounds) >= 2:
        composer = BoundComposer(
            strategy=CompositionStrategy.WORST_CASE,
            tolerance=1e-8,
        )

        # Build separator info
        sep_info: List[SeparatorInfo] = []
        for i in range(len(subgraph_bounds) - 1):
            shared = set(subgraph_bounds[i].separator_vars) & set(
                subgraph_bounds[i + 1].separator_vars
            )
            if shared:
                sep_info.append(SeparatorInfo(
                    separator_id=i,
                    variable_indices=list(shared)[:3],
                    adjacent_subgraphs=[
                        subgraph_bounds[i].subgraph_id,
                        subgraph_bounds[i + 1].subgraph_id,
                    ],
                ))

        n_sg = len(subgraph_bounds)
        overlap_mat = np.eye(n_sg)
        for i in range(n_sg - 1):
            overlap_mat[i, i + 1] = overlap_mat[i + 1, i] = 0.1

        composition_result = composer.compose(
            subgraph_bounds, sep_info,
            OverlapStructure(n_subgraphs=n_sg, overlap_matrix=overlap_mat),
        )

        print(f"    Global lower   : {composition_result.global_lower}")
        print(f"    Global upper   : {composition_result.global_upper}")
        print(f"    Gap            : {composition_result.composition_gap:.8f}")
        print(f"    Converged      : {composition_result.converged}")
    else:
        print(f"    Only {len(subgraph_bounds)} subgraph(s) — skipping composition.")

    return subgraph_bounds, composition_result


# =========================================================================
# 4. MCTS adversarial search
# =========================================================================

def run_crisis_mcts(crisis: CrisisTopology) -> Any:
    """Search for the worst-case shock scenario on the GFC network."""
    separator_line("4. MCTS Adversarial Search on GFC Network")

    G = crisis.graph
    all_nodes = list(G.nodes())

    config = SearchConfig(
        n_rollouts=500,
        budget_seconds=60.0,
        exploration_constant=1.414,
        maximize=True,
        enable_pruning=True,
        values_per_variable=6,
        shock_range=(-2.0, 2.0),
    )

    searcher = MCTSSearch(config=config, random_seed=SEED)
    engine = JunctionTreeEngine(default_bins=10)

    interface_vars = [f"N{n}" for n in all_nodes[:8]]

    result = searcher.search(
        interface_vars=interface_vars,
        inference_engine=engine,
        n_rollouts=500,
        budget_seconds=60.0,
        target_variable=interface_vars[0],
    )

    print(f"  Total rollouts   : {result.total_rollouts}")
    print(f"  Elapsed          : {result.elapsed_seconds:.2f}s")
    print(f"  Converged        : {result.converged}")

    if result.best_scenario:
        print(f"\n  Best discovered scenario:")
        print(f"    Value          : {result.best_scenario.value:.6f}")
        print(f"    Visit count    : {result.best_scenario.visit_count}")

        # Show which nodes are most shocked
        sorted_shocks = sorted(
            result.best_scenario.state.items(),
            key=lambda kv: -abs(kv[1]),
        )
        print(f"    Top shocks:")
        for var, level in sorted_shocks[:5]:
            print(f"      {var}: {level:.4f}")

    # Show top-5 scenarios
    if result.all_scenarios:
        print(f"\n  Top-5 scenarios:")
        for sc in result.all_scenarios[:5]:
            top_var = max(sc.state.items(), key=lambda kv: abs(kv[1]))
            print(f"    Rank {sc.rank}: value={sc.value:.4f}, "
                  f"top_shock={top_var[0]}={top_var[1]:.2f}")

    return result


# =========================================================================
# 5. Compare with known crisis mechanisms
# =========================================================================

def compare_with_known_crises(
    crisis: CrisisTopology,
    mcts_result,
    subgraph_bounds: List[SubgraphBound],
) -> None:
    """Check whether the discovered scenarios match documented mechanisms."""
    separator_line("5. Comparison with Known Crisis Mechanisms")

    G = crisis.graph

    # ── DebtRank baseline for known mechanisms ────────────────────────
    dr_model = DebtRankModel(variant=DebtRankVariant.LINEAR)

    print(f"  Evaluating known crisis mechanisms with DebtRank:")
    print(f"  {'Mechanism':<35s}  {'DebtRank':>10s}  {'Cascade':>10s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}")

    mechanism_impacts: Dict[str, float] = {}
    for mech in KNOWN_CRISIS_MECHANISMS:
        # Map critical node names to graph node IDs
        crisis_nodes = []
        for node_name in mech["critical_nodes"]:
            # Find the node in the graph by metadata
            for n in G.nodes():
                meta = crisis.node_metadata.get(str(n), {})
                if meta.get("name", "") == node_name or str(n) == node_name:
                    crisis_nodes.append(n)
                    break

        if not crisis_nodes:
            # Fall back to highest-degree nodes
            degrees = dict(G.degree())
            crisis_nodes = sorted(degrees, key=lambda k: -degrees[k])[:3]

        shocks = {n: 1.0 for n in crisis_nodes}
        dr_result = dr_model.compute(G, shocks, max_rounds=100)

        mechanism_impacts[mech["name"]] = dr_result.system_debtrank
        print(f"  {mech['name'][:35]:<35s}  "
              f"{dr_result.system_debtrank:>10.6f}  "
              f"{dr_result.cascade_size:>10d}")

    # ── Compare MCTS scenario with mechanisms ─────────────────────────
    if mcts_result and mcts_result.best_scenario:
        mcts_val = mcts_result.best_scenario.value
        best_mechanism = max(mechanism_impacts, key=lambda k: mechanism_impacts[k])
        best_mech_val = mechanism_impacts[best_mechanism]

        print(f"\n  MCTS discovered   : {mcts_val:.6f}")
        print(f"  Best known mech.  : {best_mech_val:.6f} ({best_mechanism})")

        if best_mech_val > 0:
            ratio = mcts_val / best_mech_val
            print(f"  Discovery ratio   : {ratio:.2f}×")
            if ratio > 1.0:
                print(f"  → MCTS found a MORE severe scenario than known mechanisms!")
            elif ratio > 0.8:
                print(f"  → MCTS found a scenario comparable to known mechanisms.")
            else:
                print(f"  → MCTS scenario is less severe than the worst known mechanism.")

    # ── Structural similarity analysis ────────────────────────────────
    print(f"\n  Structural similarity of discovered vs. known scenarios:")
    print(f"  (Measures how similar the MCTS shock pattern is to known crises)")

    if mcts_result and mcts_result.all_scenarios:
        for mech in KNOWN_CRISIS_MECHANISMS[:3]:
            # Compute overlap between MCTS-shocked nodes and known critical nodes
            known_set = set(mech["critical_nodes"])
            discovered_nodes = set()
            for var, level in mcts_result.best_scenario.state.items():
                if abs(level) > 0.5:
                    discovered_nodes.add(var)

            if known_set and discovered_nodes:
                # Rough jaccard-style similarity on node labels
                all_labels = known_set | discovered_nodes
                print(f"\n    {mech['name'][:40]}")
                print(f"      Known critical    : {known_set}")
                print(f"      MCTS discovered   : {discovered_nodes}")
                print(f"      Channel           : {mech['channel']}")


# =========================================================================
# 6. Report metrics
# =========================================================================

def report_metrics(
    crisis: CrisisTopology,
    mcts_result,
    subgraph_bounds: List[SubgraphBound],
    composition_result,
) -> None:
    """Compute and report quantitative evaluation metrics."""
    separator_line("6. Evaluation Metrics")

    metrics = MetricsComputer(
        overlap_threshold=0.5,
        overhead_acceptable_ratio=3.0,
    )

    # ── Bound quality ─────────────────────────────────────────────────
    if subgraph_bounds:
        # Average bound width across subgraphs
        widths = [float(sb.upper[0] - sb.lower[0]) for sb in subgraph_bounds]
        print(f"  Subgraph bound statistics:")
        print(f"    Count          : {len(subgraph_bounds)}")
        print(f"    Mean width     : {np.mean(widths):.6f}")
        print(f"    Median width   : {np.median(widths):.6f}")
        print(f"    Min width      : {np.min(widths):.6f}")
        print(f"    Max width      : {np.max(widths):.6f}")
        print(f"    Std dev        : {np.std(widths):.6f}")

    # ── Composition quality ───────────────────────────────────────────
    if composition_result:
        comp_width = float(composition_result.global_upper[0]
                          - composition_result.global_lower[0])
        print(f"\n  Composition bound:")
        print(f"    Width          : {comp_width:.6f}")
        print(f"    Gap            : {composition_result.composition_gap:.8f}")

    # ── Discovery ratio ───────────────────────────────────────────────
    if mcts_result and mcts_result.best_scenario:
        mcts_loss = mcts_result.best_scenario.value

        # Random baseline: just use mean of random samples
        rng = np.random.default_rng(SEED + 100)
        dr_model = DebtRankModel(variant=DebtRankVariant.LINEAR)
        random_losses = []
        G = crisis.graph
        nodes = list(G.nodes())

        for _ in range(50):
            n_shock = rng.integers(1, 4)
            shock_nodes = rng.choice(nodes, size=min(n_shock, len(nodes)),
                                     replace=False)
            shocks = {int(n): float(rng.uniform(0.3, 1.0))
                      for n in shock_nodes}
            dr_res = dr_model.compute(G, shocks, max_rounds=50)
            random_losses.append(dr_res.system_debtrank)

        baseline_loss = max(random_losses) if random_losses else 0

        discovery = metrics.compute_discovery_ratio(
            mcts_loss=mcts_loss,
            baseline_loss=baseline_loss,
        )

        print(f"\n  Discovery ratio:")
        print(f"    MCTS loss      : {discovery.mcts_loss:.6f}")
        print(f"    Baseline loss  : {discovery.baseline_loss:.6f}")
        print(f"    Ratio          : {discovery.ratio:.4f}")
        print(f"    Improvement    : {discovery.improvement_pct:.1f}%")

    # ── Network-level risk summary ────────────────────────────────────
    print(f"\n  Network risk profile:")
    G = crisis.graph
    sensitivity = dr_model.sensitivity_analysis(G, shock_level=1.0)
    print(f"    System vulnerability  : {sensitivity.system_vulnerability:.6f}")
    print(f"    Concentration risk    : {sensitivity.concentration_risk:.6f}")
    print(f"    Top-3 risky nodes:")
    for rank, (node, impact) in enumerate(sensitivity.top_k_nodes[:3], 1):
        name = crisis.node_metadata.get(str(node), {}).get("name", str(node))
        print(f"      {rank}. {name} — impact = {impact:.6f}")


# =========================================================================
# Main entry point
# =========================================================================

def main() -> None:
    """Run the full historical crisis simulation."""
    print("CausalBound — Historical Crisis Simulation (2008 GFC)")
    print("=" * 72)

    t0 = time.perf_counter()

    # Step 1: Reconstruct the GFC network
    crisis = reconstruct_gfc_network()

    # Step 2: Build SCM with CDS instruments
    scm, cds_models = build_scm_with_cds(crisis)

    # Step 3: Run the CausalBound pipeline
    subgraph_bounds, composition_result = run_causalbound_pipeline(crisis, scm)

    # Step 4: MCTS adversarial search
    mcts_result = run_crisis_mcts(crisis)

    # Step 5: Compare with known crisis mechanisms
    compare_with_known_crises(crisis, mcts_result, subgraph_bounds)

    # Step 6: Report metrics
    report_metrics(crisis, mcts_result, subgraph_bounds, composition_result)

    elapsed = time.perf_counter() - t0
    separator_line("Final Summary")
    print(f"  Crisis           : 2008 Global Financial Crisis")
    print(f"  Network          : {crisis.graph.number_of_nodes()} institutions, "
          f"{crisis.graph.number_of_edges()} exposures")
    print(f"  CDS instruments  : {len(cds_models)} modelled")
    print(f"  Subgraph bounds  : {len(subgraph_bounds)} computed")
    if mcts_result and mcts_result.best_scenario:
        print(f"  MCTS best value  : {mcts_result.best_scenario.value:.6f}")
    print(f"  Known mechanisms : {len(KNOWN_CRISIS_MECHANISMS)} evaluated")
    print(f"  Total elapsed    : {elapsed:.2f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
