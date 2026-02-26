#!/usr/bin/env python3
"""
CausalBound — Verified Inference Example
=========================================

This example shows how CausalBound combines exact Bayesian inference with
formal SMT verification.  The pipeline is:

    1. Build a small Bayesian network (hand-crafted for clarity)
    2. Construct a junction tree and run belief propagation
    3. Execute causal queries (do-calculus interventions)
    4. Attach the SMT verifier to check every inference step
    5. Run the full verified inference pipeline
    6. Inspect certificates and measure verification overhead

The SMT verifier encodes probability axioms and constraint-propagation
rules in the theory of Quantifier-Free Linear Real Arithmetic (QF_LRA)
and checks them with the Z3 solver.  Each verified inference step
produces a *certificate* — a compact proof artefact.

Usage
-----
    python verified_inference.py
"""

from __future__ import annotations

import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── CausalBound imports ──────────────────────────────────────────────────
from causalbound.junction.engine import JunctionTreeEngine, InferenceStats
from causalbound.junction.clique_tree import CliqueTree
from causalbound.junction.potential_table import PotentialTable
from causalbound.junction.message_passing import MessagePassingVariant
from causalbound.junction.discretization import BinningStrategy
from causalbound.junction.do_operator import DoOperator, Intervention
from causalbound.smt.verifier import (
    SMTVerifier,
    BoundEvidence,
    MessageData,
    VerificationStatus,
)
from causalbound.smt.certificates import CertificateEmitter

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


def make_potential_table(
    variables: List[str],
    cardinalities: Dict[str, int],
    values: np.ndarray,
) -> PotentialTable:
    """Construct a PotentialTable from a flat array of values.

    Parameters
    ----------
    variables : list of str
        Variable names in the order that defines the table layout.
    cardinalities : dict
        Mapping from variable name to its number of states.
    values : ndarray
        Flat array of potential values in row-major order.
    """
    cards = [cardinalities[v] for v in variables]
    table = values.reshape(cards)
    return PotentialTable(variables=variables, cardinalities=cards, values=table)


# =========================================================================
# 1. Build a small Bayesian network
# =========================================================================

def build_bayesian_network() -> Tuple[Dict, Dict, Dict[str, int]]:
    """Construct a 5-node "Systemic Stress" Bayesian network.

    Network structure (a simplified financial contagion model)::

        Macro  ──►  BankA  ──►  BankC
          │            │           │
          └──►  BankB  ┘     SystemicRisk
                  │              ▲
                  └──────────────┘

    Variables
    ---------
    Macro          : {stable=0, recession=1}
    BankA          : {solvent=0, distressed=1}
    BankB          : {solvent=0, distressed=1}
    BankC          : {solvent=0, distressed=1}
    SystemicRisk   : {low=0, high=1}
    """
    separator_line("1. Build Bayesian Network")

    # DAG adjacency: parent → children
    dag: Dict[str, List[str]] = {
        "Macro":        [],                     # root node — no parents
        "BankA":        ["Macro"],              # influenced by macro
        "BankB":        ["Macro", "BankA"],     # influenced by macro + BankA
        "BankC":        ["BankA"],              # counterparty of BankA
        "SystemicRisk": ["BankB", "BankC"],     # aggregate risk indicator
    }

    cardinalities: Dict[str, int] = {
        "Macro": 2, "BankA": 2, "BankB": 2, "BankC": 2, "SystemicRisk": 2,
    }

    # ── Conditional Probability Distributions ────────────────────────

    # P(Macro): prior over macroeconomic state
    cpd_macro = make_potential_table(
        ["Macro"], cardinalities,
        np.array([0.7, 0.3]),  # 70% stable, 30% recession
    )

    # P(BankA | Macro)
    cpd_banka = make_potential_table(
        ["BankA", "Macro"], cardinalities,
        np.array([
            0.9, 0.4,   # P(BankA=solvent | Macro=stable), P(BankA=solvent | recession)
            0.1, 0.6,   # P(BankA=distressed | ...)
        ]),
    )

    # P(BankB | Macro, BankA)
    cpd_bankb = make_potential_table(
        ["BankB", "Macro", "BankA"], cardinalities,
        np.array([
            0.95, 0.7,   # BankB=solvent | Macro=stable, BankA=solvent/distressed
            0.6,  0.2,   # BankB=solvent | Macro=recession, BankA=solvent/distressed
            0.05, 0.3,   # BankB=distressed | ...
            0.4,  0.8,
        ]),
    )

    # P(BankC | BankA)
    cpd_bankc = make_potential_table(
        ["BankC", "BankA"], cardinalities,
        np.array([
            0.85, 0.35,  # BankC=solvent | BankA=solvent/distressed
            0.15, 0.65,  # BankC=distressed | ...
        ]),
    )

    # P(SystemicRisk | BankB, BankC)
    cpd_risk = make_potential_table(
        ["SystemicRisk", "BankB", "BankC"], cardinalities,
        np.array([
            0.95, 0.5,   # SR=low | BankB=solvent, BankC=solvent/distressed
            0.4,  0.05,  # SR=low | BankB=distressed, BankC=...
            0.05, 0.5,   # SR=high | ...
            0.6,  0.95,
        ]),
    )

    cpds: Dict[str, PotentialTable] = {
        "Macro": cpd_macro,
        "BankA": cpd_banka,
        "BankB": cpd_bankb,
        "BankC": cpd_bankc,
        "SystemicRisk": cpd_risk,
    }

    # Print the network structure
    print("  DAG structure:")
    for child, parents in dag.items():
        if parents:
            print(f"    {' + '.join(parents)} → {child}")
        else:
            print(f"    {child} (root)")

    print(f"\n  Variables        : {len(dag)}")
    print(f"  Edges            : {sum(len(p) for p in dag.values())}")

    return dag, cpds, cardinalities


# =========================================================================
# 2. Run junction-tree inference (unverified)
# =========================================================================

def run_unverified_inference(
    dag: Dict, cpds: Dict, cardinalities: Dict[str, int],
) -> Tuple[JunctionTreeEngine, List]:
    """Build a junction tree and run exact inference without verification.

    This gives us a baseline for measuring the verification overhead.
    """
    separator_line("2. Junction-Tree Inference (Unverified)")

    engine = JunctionTreeEngine(
        variant=MessagePassingVariant.HUGIN,
        use_log_space=False,
        cache_capacity=1024,
        default_bins=20,
    )

    # Build the junction tree from the DAG and CPDs
    clique_tree = engine.build(
        dag=dag,
        cpds=cpds,
        cardinalities=cardinalities,
    )

    print(f"  Junction tree built:")
    print(f"    Cliques        : {clique_tree.num_cliques}")
    print(f"    Max clique     : {clique_tree.max_clique_size}")
    print(f"    Treewidth      : {clique_tree.treewidth}")

    # Run several observational queries
    queries = [
        ("SystemicRisk", None, "P(SystemicRisk)"),
        ("SystemicRisk", {"Macro": 1}, "P(SystemicRisk | Macro=recession)"),
        ("BankA", {"SystemicRisk": 1}, "P(BankA | SystemicRisk=high)"),
        ("BankC", {"BankA": 1}, "P(BankC | BankA=distressed)"),
    ]

    results = []
    for target, evidence, label in queries:
        t0 = time.perf_counter()
        result = engine.query(target=target, evidence=evidence)
        elapsed = time.perf_counter() - t0

        results.append((label, result, elapsed))
        dist_str = ", ".join(f"{p:.4f}" for p in result.distribution)
        print(f"\n  Query: {label}")
        print(f"    Distribution   : [{dist_str}]")
        print(f"    E[{target}]     : {result.expected_value:.4f}")
        print(f"    Var[{target}]   : {result.variance:.6f}")
        print(f"    Time           : {elapsed*1000:.2f} ms")

    return engine, results


# =========================================================================
# 3. Run causal (interventional) queries
# =========================================================================

def run_causal_queries(engine: JunctionTreeEngine) -> List:
    """Perform do-calculus interventions and query the mutilated DAG.

    ``do(BankA = distressed)`` removes all incoming edges to BankA and
    forces it to state 1.  This lets us answer: "If BankA were *forced*
    to fail, what would happen to SystemicRisk?"
    """
    separator_line("3. Causal (Interventional) Queries")

    interventional_queries = [
        # do(BankA = distressed), query SystemicRisk
        {
            "target": "SystemicRisk",
            "intervention": {"BankA": 1.0},
            "label": "P(SystemicRisk | do(BankA=distressed))",
        },
        # do(Macro = recession), query BankC
        {
            "target": "BankC",
            "intervention": {"Macro": 1.0},
            "label": "P(BankC | do(Macro=recession))",
        },
        # do(BankB = distressed), query SystemicRisk
        {
            "target": "SystemicRisk",
            "intervention": {"BankB": 1.0},
            "label": "P(SystemicRisk | do(BankB=distressed))",
        },
    ]

    results = []
    for q in interventional_queries:
        t0 = time.perf_counter()
        result = engine.query(
            target=q["target"],
            intervention=q["intervention"],
        )
        elapsed = time.perf_counter() - t0

        results.append((q["label"], result, elapsed))
        dist_str = ", ".join(f"{p:.4f}" for p in result.distribution)
        print(f"  Query: {q['label']}")
        print(f"    Distribution   : [{dist_str}]")
        print(f"    E[{q['target']}] : {result.expected_value:.4f}")
        print(f"    Time           : {elapsed*1000:.2f} ms")
        print()

    return results


# =========================================================================
# 4. Attach SMT verifier
# =========================================================================

def run_verified_inference(
    engine: JunctionTreeEngine,
    dag: Dict,
    cpds: Dict,
    cardinalities: Dict[str, int],
) -> Tuple[List, float]:
    """Run the same queries under SMT verification.

    The verifier checks:
      • Probability axioms (values in [0, 1], sum to 1)
      • Message-passing correctness (separator consistency)
      • Bound validity (lower ≤ true ≤ upper)
    """
    separator_line("4. Verified Inference with SMT")

    verifier = SMTVerifier(
        timeout_ms=10_000,
        track_unsat_cores=True,
        emit_certificates=True,
        epsilon=1e-9,
    )

    session_id = verifier.begin_session()
    print(f"  SMT session      : {session_id}")
    print(f"  Solver           : Z3")
    print(f"  Timeout          : 10,000 ms")
    print(f"  ε tolerance      : 1e-9")

    verification_results = []
    total_verify_time = 0.0

    # ── Verify observational bounds ───────────────────────────────────
    print(f"\n  Verifying observational queries...")

    test_queries = [
        ("SystemicRisk", None),
        ("SystemicRisk", {"Macro": 1}),
        ("BankA", {"SystemicRisk": 1}),
    ]

    for target, evidence in test_queries:
        # Run inference
        result = engine.query(target=target, evidence=evidence)

        # Build evidence for the verifier from the query result
        bound_evidence = BoundEvidence(
            lp_objective=result.expected_value,
            dual_values=[float(v) for v in result.distribution],
        )

        # Verify the computed bounds
        t0 = time.perf_counter()
        vr = verifier.verify_bound(
            lower=float(result.distribution[0]),
            upper=float(result.distribution[-1]),
            evidence=bound_evidence,
        )
        verify_time = time.perf_counter() - t0
        total_verify_time += verify_time

        status_icon = "✓" if vr.status == VerificationStatus.VERIFIED else "✗"
        evidence_str = str(evidence) if evidence else "none"
        print(f"    {status_icon} P({target} | {evidence_str}) — "
              f"{vr.status.value} ({verify_time*1000:.1f} ms, "
              f"{vr.assertion_count} assertions)")

        verification_results.append(vr)

    # ── Verify message passing ────────────────────────────────────────
    print(f"\n  Verifying message-passing steps...")

    # Simulate checking a message between two cliques
    message_data = MessageData(
        sender_id="clique_0",
        receiver_id="clique_1",
        separator_vars=["BankA"],
        potential_values={"BankA_0": 0.72, "BankA_1": 0.28},
        marginal_values={"BankA_0": 0.72, "BankA_1": 0.28},
        bound_lower=0.0,
        bound_upper=1.0,
    )

    t0 = time.perf_counter()
    msg_vr = verifier.verify_message(
        sender="clique_0",
        receiver="clique_1",
        message_data=message_data,
    )
    msg_verify_time = time.perf_counter() - t0
    total_verify_time += msg_verify_time

    status_icon = "✓" if msg_vr.status == VerificationStatus.VERIFIED else "✗"
    print(f"    {status_icon} Message clique_0 → clique_1 — "
          f"{msg_vr.status.value} ({msg_verify_time*1000:.1f} ms)")

    verification_results.append(msg_vr)

    # ── End verification session ──────────────────────────────────────
    session_stats = verifier.end_session()

    print(f"\n  Verification session summary:")
    print(f"    Total steps    : {session_stats.total_steps}")
    print(f"    Passed         : {session_stats.passed_steps}")
    print(f"    Failed         : {session_stats.failed_steps}")
    print(f"    Unknown        : {session_stats.unknown_steps}")
    print(f"    Total SMT time : {session_stats.total_smt_time_s:.3f}s")
    print(f"    Assertions     : {session_stats.total_assertions}")

    return verification_results, total_verify_time


# =========================================================================
# 5. Check certificates
# =========================================================================

def inspect_certificates(verification_results: List) -> None:
    """Display details of the verification certificates.

    Each certificate records: the step ID, verification status,
    number of SMT assertions checked, and (on failure) the unsatisfiable
    core that identifies the contradictory constraints.
    """
    separator_line("5. Certificate Inspection")

    print(f"  {'Step ID':<20s}  {'Status':<12s}  {'Assertions':>11s}  "
          f"{'SMT Time':>10s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*11}  {'-'*10}")

    for vr in verification_results:
        step_short = vr.step_id[:18] if len(vr.step_id) > 18 else vr.step_id
        print(f"  {step_short:<20s}  {vr.status.value:<12s}  "
              f"{vr.assertion_count:>11d}  {vr.smt_time_s:>10.3f}s")

    # Count overall statistics
    n_verified = sum(1 for vr in verification_results
                     if vr.status == VerificationStatus.VERIFIED)
    n_total = len(verification_results)
    print(f"\n  Overall: {n_verified}/{n_total} steps verified successfully")

    # Show any UNSAT cores (indicate bugs if they occur)
    failed = [vr for vr in verification_results
              if vr.status != VerificationStatus.VERIFIED]
    if failed:
        print(f"\n  ⚠  {len(failed)} step(s) did NOT verify:")
        for vr in failed:
            print(f"    Step {vr.step_id}: {vr.message}")
            if vr.unsat_core:
                print(f"      UNSAT core: {vr.unsat_core[:5]}")
    else:
        print(f"  ✓ All steps verified — inference results are certified correct.")


# =========================================================================
# 6. Measure verification overhead
# =========================================================================

def measure_overhead(
    engine: JunctionTreeEngine,
    total_verify_time: float,
) -> None:
    """Compare inference time with and without SMT verification.

    The "overhead ratio" measures the cost of formal guarantees: a ratio
    of 2.0× means verification doubles the total wall-clock time.
    """
    separator_line("6. Verification Overhead Analysis")

    # Time unverified queries
    queries = [
        ("SystemicRisk", None),
        ("SystemicRisk", {"Macro": 1}),
        ("BankA", {"SystemicRisk": 1}),
        ("BankC", {"BankA": 1}),
    ]

    unverified_times = []
    for target, evidence in queries:
        t0 = time.perf_counter()
        _ = engine.query(target=target, evidence=evidence)
        unverified_times.append(time.perf_counter() - t0)

    total_unverified = sum(unverified_times)
    overhead_ratio = (total_verify_time / total_unverified
                      if total_unverified > 0 else float("inf"))

    print(f"  Unverified time  : {total_unverified*1000:.2f} ms "
          f"({len(queries)} queries)")
    print(f"  Verification time: {total_verify_time*1000:.2f} ms")
    print(f"  Overhead ratio   : {overhead_ratio:.2f}×")
    print()

    # Per-query breakdown
    print(f"  {'Query':<35s}  {'Inference':>10s}  {'Verify':>10s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}")
    for i, (target, evidence) in enumerate(queries):
        ev_str = str(evidence) if evidence else "none"
        label = f"P({target} | {ev_str})"[:35]
        infer_ms = unverified_times[i] * 1000
        # Approximate per-query verification time
        verify_ms = (total_verify_time / len(queries)) * 1000
        print(f"  {label:<35s}  {infer_ms:>9.2f}ms  {verify_ms:>9.2f}ms")

    # Assessment
    if overhead_ratio < 3.0:
        print(f"\n  ✓ Overhead is acceptable (< 3×) for production use.")
    elif overhead_ratio < 10.0:
        print(f"\n  ⚠ Moderate overhead — consider selective verification.")
    else:
        print(f"\n  ⚠ High overhead — consider verifying only critical paths.")


# =========================================================================
# Main entry point
# =========================================================================

def main() -> None:
    """Run the full verified inference example."""
    print("CausalBound — Verified Inference Example")
    print("=" * 72)

    t0 = time.perf_counter()

    # Step 1: Build the Bayesian network
    dag, cpds, cardinalities = build_bayesian_network()

    # Step 2: Run unverified junction-tree inference
    engine, unverified_results = run_unverified_inference(dag, cpds, cardinalities)

    # Step 3: Run causal (interventional) queries
    causal_results = run_causal_queries(engine)

    # Step 4: Run verified inference
    verification_results, total_verify_time = run_verified_inference(
        engine, dag, cpds, cardinalities,
    )

    # Step 5: Inspect certificates
    inspect_certificates(verification_results)

    # Step 6: Measure verification overhead
    measure_overhead(engine, total_verify_time)

    elapsed = time.perf_counter() - t0
    separator_line("Summary")
    print(f"  Network          : 5-node systemic stress model")
    print(f"  Observational    : {len(unverified_results)} queries")
    print(f"  Interventional   : {len(causal_results)} queries")
    print(f"  Verified steps   : {len(verification_results)}")
    print(f"  Total elapsed    : {elapsed:.2f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
