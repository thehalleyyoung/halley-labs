#!/usr/bin/env python3
"""
Leader Election — CoaCert-TLA Demonstration
=============================================

Demonstrates the CoaCert pipeline on a ring-based Leader Election
protocol (Chang-Roberts) with N=3 nodes:

  1. Build specification and validate
  2. Explore state space
  3. Compress via bisimulation
  4. Generate and verify witness
  5. Check leader uniqueness and eventual election properties

Usage:
    python -m examples.leader_election_demo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Imports with error handling
# ---------------------------------------------------------------------------

try:
    from coacert.specs import LeaderElectionSpec, SpecRegistry
    from coacert.parser import PrettyPrinter
    from coacert.explorer import ExplicitStateExplorer, TransitionGraph
    from coacert.bisimulation import (
        RefinementEngine,
        RefinementStrategy,
        QuotientBuilder,
    )
    from coacert.witness import (
        WitnessSet,
        TransitionWitness,
        EquivalenceBinding,
        HashChain,
        WitnessFormat,
    )
    from coacert.verifier import verify_witness, Verdict
    from coacert.properties import (
        SafetyChecker,
        KripkeAdapter,
        AG,
        Atomic,
    )
    from coacert.evaluation import Timer, format_duration
except ImportError as exc:
    print(f"ERROR: Could not import coacert: {exc}")
    print("Run from the implementation/ directory or install coacert.")
    sys.exit(1)


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   CoaCert-TLA: Leader Election Protocol Demonstration      ║
╚══════════════════════════════════════════════════════════════╝
"""

N_NODES = 3
WITNESS_PATH = Path("leader_election_witness.cwit")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def build_spec(n: int) -> LeaderElectionSpec:
    """Build and validate the Leader Election specification."""
    print(f"\n  [1/5] Building Leader Election spec (N={n})...")
    spec = LeaderElectionSpec(n_nodes=n)
    module = spec.get_spec()
    properties = spec.get_properties()
    config = spec.get_config()

    errors = spec.validate()
    print(f"        Module: {module.name}")
    print(f"        Variables: {sum(len(v.names) for v in module.variables)}")
    print(f"        Definitions: {len(module.definitions)}")
    print(f"        Properties: {len(properties)}")
    for p in properties:
        print(f"          • {p.name}")
    if errors:
        print(f"        ⚠ Errors: {errors}")
    else:
        print(f"        ✓ Valid")

    # Show config highlights
    est = config.get("expected_states", {})
    ub = est.get("upper_bound", "?")
    print(f"        State-space upper bound: {ub:,}" if isinstance(ub, int)
          else f"        State-space upper bound: {ub}")
    return spec


def explore_states(spec: LeaderElectionSpec) -> TransitionGraph:
    """Explore the full reachable state space."""
    print(f"\n  [2/5] Exploring state space...")
    module = spec.get_spec()
    config = spec.get_config()

    from coacert.semantics import ActionEvaluator, Environment
    env = Environment()
    env.bind_constants(config["constants"])
    engine = ActionEvaluator(module, env)

    timer = Timer()
    timer.start()
    explorer = ExplicitStateExplorer(engine, depth_limit=80)
    graph = explorer.explore()
    stats = explorer.stats
    timer.stop()

    print(f"        States:      {stats.states_explored:,}")
    print(f"        Transitions: {stats.transitions_found:,}")
    print(f"        Max depth:   {stats.max_depth_reached}")
    print(f"        Time:        {format_duration(timer.elapsed)}")
    print(f"        ✓ Done")
    return graph


def compress(graph: TransitionGraph):
    """Compute bisimulation quotient."""
    print(f"\n  [3/5] Computing bisimulation quotient...")
    timer = Timer()
    timer.start()

    engine = RefinementEngine(graph, strategy=RefinementStrategy.EAGER)
    result = engine.run()

    quotient = QuotientBuilder(result.partition, graph).build()
    timer.stop()

    orig = graph.state_count
    quot = quotient.state_count
    ratio = orig / quot if quot else float("inf")

    print(f"        Rounds:          {result.rounds}")
    print(f"        Original states: {orig:,}")
    print(f"        Quotient states: {quot:,}")
    print(f"        Compression:     {ratio:.2f}x")
    print(f"        Time:            {format_duration(timer.elapsed)}")
    print(f"        ✓ Done")
    return result, quotient


def generate_and_verify(result, graph, quotient):
    """Generate witness and verify it."""
    print(f"\n  [4/5] Generating & verifying witness...")
    timer = Timer()
    timer.start()

    binding = EquivalenceBinding()
    for sid in graph.state_ids:
        cid = result.partition.get_class(sid)
        binding.bind(sid, f"class_{cid}")

    ws = WitnessSet()
    for edge in quotient.edges:
        ws.add_transition(TransitionWitness(
            source_class=f"class_{edge.source}",
            target_class=f"class_{edge.target}",
            action=edge.label,
            concrete_source=edge.source,
            concrete_target=edge.target,
        ))

    chain = HashChain()
    chain.add_equivalence_block(binding)
    chain.add_transition_block(ws)

    witness_fmt = WitnessFormat(
        equivalence=binding, witnesses=ws, chain=chain,
        original_state_count=graph.state_count,
        quotient_state_count=quotient.state_count,
    )
    nbytes = witness_fmt.serialize(WITNESS_PATH)

    report = verify_witness(str(WITNESS_PATH))
    timer.stop()

    print(f"        Witnesses:   {ws.total_count}")
    print(f"        Merkle root: {ws.root.hex()[:24]}...")
    print(f"        File size:   {nbytes:,} bytes")
    print(f"        Verdict:     {report.verdict.name}")
    print(f"        Time:        {format_duration(timer.elapsed)}")
    if report.verdict == Verdict.PASS:
        print(f"        ✓ Witness verified")
    else:
        print(f"        ✗ Verification failed")
        for err in report.errors[:3]:
            print(f"          {err}")
    return report


def check_properties(quotient, spec: LeaderElectionSpec):
    """Check leader uniqueness and eventual election properties."""
    print(f"\n  [5/5] Checking properties...")
    timer = Timer()
    timer.start()

    kripke = KripkeAdapter(quotient)
    properties = spec.get_properties()

    results = []
    for prop in properties:
        checker = SafetyChecker(kripke)
        cr = checker.check(prop)
        status = "✓" if cr.holds else "✗"
        results.append({"name": prop.name, "holds": cr.holds})
        print(f"        {status} {prop.name}")

    timer.stop()
    passed = sum(1 for r in results if r["holds"])
    print(f"        Results: {passed}/{len(results)} pass")
    print(f"        Time:    {format_duration(timer.elapsed)}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(BANNER)
    total_start = time.time()

    try:
        spec = build_spec(N_NODES)
        graph = explore_states(spec)
        result, quotient = compress(graph)
        report = generate_and_verify(result, graph, quotient)
        prop_results = check_properties(quotient, spec)

        total = time.time() - total_start
        ratio = graph.state_count / quotient.state_count \
            if quotient.state_count else float("inf")
        passed = sum(1 for r in prop_results if r["holds"])

        print(f"\n{'─'*60}")
        print(f"  Leader Election (N={N_NODES}) — Summary")
        print(f"{'─'*60}")
        print(f"  States:      {graph.state_count:,} → {quotient.state_count:,}"
              f" ({ratio:.1f}x compression)")
        print(f"  Witness:     {report.verdict.name}")
        print(f"  Properties:  {passed}/{len(prop_results)} pass")
        print(f"  Total time:  {format_duration(total)}")
        print()

        return 0 if report.verdict == Verdict.PASS else 1

    except Exception as exc:
        print(f"\n  ✗ Failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if WITNESS_PATH.exists():
            WITNESS_PATH.unlink()


if __name__ == "__main__":
    sys.exit(main())
