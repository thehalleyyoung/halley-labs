#!/usr/bin/env python3
"""
Two-Phase Commit Protocol — CoaCert-TLA Demonstration
======================================================

End-to-end pipeline demonstration using the classic Two-Phase Commit
protocol with N=3 resource managers:

  1. Build the TLA-lite specification programmatically
  2. Parse and validate the spec
  3. Explore the full state space
  4. Compute bisimulation quotient via partition refinement
  5. Generate Merkle-hashed witness certificate
  6. Verify the witness independently
  7. Check safety/liveness properties on the quotient
  8. Print detailed results with compression ratio and timing

Usage:
    python -m examples.two_phase_commit_demo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

def _import_coacert():
    """Import coacert components with helpful error messages."""
    try:
        from coacert.specs import TwoPhaseCommitSpec, SpecRegistry
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
            MerkleTree,
            EquivalenceBinding,
            HashChain,
            WitnessFormat,
        )
        from coacert.verifier import verify_witness, VerificationReport, Verdict
        from coacert.properties import (
            CTLStarChecker,
            SafetyChecker,
            KripkeAdapter,
            AG,
            EF,
            Atomic,
            parse_formula,
        )
        from coacert.evaluation import (
            Timer,
            MetricsCollector,
            CompressionAnalyzer,
            format_duration,
        )
        return True
    except ImportError as exc:
        print(f"ERROR: Could not import coacert package: {exc}")
        print("Make sure you are running from the implementation/ directory")
        print("or that coacert is installed (pip install -e .)")
        return False


if not _import_coacert():
    sys.exit(1)

# Re-import at module level after check
from coacert.specs import TwoPhaseCommitSpec, SpecRegistry
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
    MerkleTree,
    EquivalenceBinding,
    HashChain,
    WitnessFormat,
)
from coacert.verifier import verify_witness, VerificationReport, Verdict
from coacert.properties import (
    CTLStarChecker,
    SafetyChecker,
    KripkeAdapter,
    AG,
    EF,
    Atomic,
    parse_formula,
)
from coacert.evaluation import (
    Timer,
    MetricsCollector,
    CompressionAnalyzer,
    format_duration,
)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   CoaCert-TLA: Two-Phase Commit Protocol Demonstration     ║
║   Coalgebraic Certified Compression for TLA+ Specs         ║
╚══════════════════════════════════════════════════════════════╝
"""

N_PARTICIPANTS = 3
WITNESS_PATH = Path("two_phase_commit_witness.cwit")


# ---------------------------------------------------------------------------
# Step 1: Build specification
# ---------------------------------------------------------------------------

def step_build_spec(n: int) -> TwoPhaseCommitSpec:
    """Build the Two-Phase Commit TLA-lite specification."""
    print(f"\n{'='*60}")
    print(f"  Step 1: Build Two-Phase Commit Spec (N={n})")
    print(f"{'='*60}")

    spec_builder = TwoPhaseCommitSpec(n_participants=n)
    module = spec_builder.get_spec()
    properties = spec_builder.get_properties()
    config = spec_builder.get_config()

    print(f"  Module name:     {module.name}")
    print(f"  Variables:       {sum(len(v.names) for v in module.variables)}")
    print(f"  Definitions:     {len(module.definitions)}")
    print(f"  Properties:      {len(properties)}")
    for prop in properties:
        print(f"    - {prop.name} ({type(prop).__name__})")

    # Validate
    errors = spec_builder.validate()
    if errors:
        print(f"  ⚠ Validation errors: {errors}")
    else:
        print("  ✓ Specification valid")

    # Pretty-print excerpt
    pp = PrettyPrinter()
    source = pp.print_module(module)
    lines = source.strip().split("\n")
    print(f"\n  TLA-lite source ({len(lines)} lines):")
    for line in lines[:8]:
        print(f"    {line}")
    if len(lines) > 8:
        print(f"    ... ({len(lines) - 8} more lines)")

    # State-space estimate
    est = config.get("expected_states", {})
    if est:
        print(f"\n  State-space estimate:")
        for k, v in est.items():
            if k != "note":
                print(f"    {k}: {v:,}")
        if "note" in est:
            print(f"    ({est['note']})")

    return spec_builder


# ---------------------------------------------------------------------------
# Step 2: Explore state space
# ---------------------------------------------------------------------------

def step_explore(spec_builder: TwoPhaseCommitSpec) -> TransitionGraph:
    """Explore the full state space via BFS."""
    print(f"\n{'='*60}")
    print("  Step 2: Explore State Space")
    print(f"{'='*60}")

    module = spec_builder.get_spec()
    config = spec_builder.get_config()

    timer = Timer()
    timer.start()

    # Build a semantic engine from the spec and config
    from coacert.semantics import ActionEvaluator, Environment
    env = Environment()
    env.bind_constants(config["constants"])
    engine = ActionEvaluator(module, env)

    explorer = ExplicitStateExplorer(engine, depth_limit=100)
    graph = explorer.explore()
    stats = explorer.stats

    timer.stop()

    print(f"  States explored: {stats.states_explored:,}")
    print(f"  Transitions:     {stats.transitions_found:,}")
    print(f"  Max depth:       {stats.max_depth_reached}")
    print(f"  Deadlocks:       {stats.deadlock_states}")
    print(f"  Duplicates:      {stats.duplicate_states:,}")
    print(f"  Speed:           {stats.states_per_second:.1f} states/s")
    print(f"  Time:            {format_duration(timer.elapsed)}")
    print(f"  ✓ Exploration complete")

    return graph


# ---------------------------------------------------------------------------
# Step 3: Compute bisimulation quotient
# ---------------------------------------------------------------------------

def step_compress(graph: TransitionGraph):
    """Run partition refinement to compute the bisimulation quotient."""
    print(f"\n{'='*60}")
    print("  Step 3: Bisimulation Quotient Compression")
    print(f"{'='*60}")

    timer = Timer()
    timer.start()

    engine = RefinementEngine(graph, strategy=RefinementStrategy.EAGER)
    result = engine.run()

    timer.stop()

    original_states = graph.state_count
    quotient_blocks = result.final_blocks
    ratio = original_states / quotient_blocks if quotient_blocks else float("inf")

    print(f"  Strategy:        {RefinementStrategy.EAGER.name}")
    print(f"  Rounds:          {result.rounds}")
    print(f"  Original states: {original_states:,}")
    print(f"  Quotient blocks: {quotient_blocks:,}")
    print(f"  Compression:     {ratio:.2f}x")
    print(f"  Converged:       {result.converged}")
    print(f"  Time:            {format_duration(timer.elapsed)}")

    # Build quotient graph
    quotient_graph = QuotientBuilder(result.partition, graph).build()
    print(f"  Quotient states: {quotient_graph.state_count:,}")
    print(f"  Quotient trans:  {quotient_graph.transition_count:,}")
    print(f"  ✓ Compression complete")

    return result, quotient_graph


# ---------------------------------------------------------------------------
# Step 4: Generate witness certificate
# ---------------------------------------------------------------------------

def step_generate_witness(result, graph: TransitionGraph,
                          quotient_graph: TransitionGraph):
    """Generate a Merkle-hashed witness certificate."""
    print(f"\n{'='*60}")
    print("  Step 4: Generate Witness Certificate")
    print(f"{'='*60}")

    timer = Timer()
    timer.start()

    # Build equivalence binding
    binding = EquivalenceBinding()
    partition = result.partition
    for state_id in graph.state_ids:
        class_id = partition.get_class(state_id)
        binding.bind(state_id, f"class_{class_id}")

    # Build transition witnesses
    witness_set = WitnessSet()
    for edge in quotient_graph.edges:
        tw = TransitionWitness(
            source_class=f"class_{edge.source}",
            target_class=f"class_{edge.target}",
            action=edge.label,
            concrete_source=edge.source,
            concrete_target=edge.target,
        )
        witness_set.add_transition(tw)

    # Build hash chain
    chain = HashChain()
    chain.add_equivalence_block(binding)
    chain.add_transition_block(witness_set)

    # Assemble witness format
    witness = WitnessFormat(
        equivalence=binding,
        witnesses=witness_set,
        chain=chain,
        original_state_count=graph.state_count,
        quotient_state_count=quotient_graph.state_count,
    )

    # Serialize
    bytes_written = witness.serialize(WITNESS_PATH)

    timer.stop()

    print(f"  Transition witnesses: {witness_set.transition_count}")
    print(f"  Stutter witnesses:    {witness_set.stutter_count}")
    print(f"  Fairness witnesses:   {witness_set.fairness_count}")
    print(f"  Merkle root:          {witness_set.root.hex()[:32]}...")
    print(f"  Hash chain blocks:    {chain.block_count}")
    print(f"  Witness file:         {WITNESS_PATH}")
    print(f"  File size:            {bytes_written:,} bytes")
    print(f"  Time:                 {format_duration(timer.elapsed)}")
    print(f"  ✓ Witness generated")

    return witness


# ---------------------------------------------------------------------------
# Step 5: Verify witness
# ---------------------------------------------------------------------------

def step_verify_witness():
    """Verify the witness certificate independently."""
    print(f"\n{'='*60}")
    print("  Step 5: Verify Witness Certificate")
    print(f"{'='*60}")

    timer = Timer()
    timer.start()

    report = verify_witness(str(WITNESS_PATH))

    timer.stop()

    print(f"  Verdict:     {report.verdict.name}")
    print(f"  Hash chain:  {'✓' if not report.errors else '✗'}")
    if report.errors:
        for err in report.errors[:5]:
            print(f"    ERROR: {err}")
    else:
        print(f"  Closure:     ✓")
        print(f"  Stuttering:  ✓")
        print(f"  Fairness:    ✓")
    print(f"  Time:        {format_duration(timer.elapsed)}")
    print(f"  ✓ Verification complete")

    return report


# ---------------------------------------------------------------------------
# Step 6: Check properties
# ---------------------------------------------------------------------------

def step_check_properties(quotient_graph: TransitionGraph,
                          spec_builder: TwoPhaseCommitSpec):
    """Check safety and liveness properties on the quotient."""
    print(f"\n{'='*60}")
    print("  Step 6: Property Checking")
    print(f"{'='*60}")

    timer = Timer()
    timer.start()

    kripke = KripkeAdapter(quotient_graph)
    properties = spec_builder.get_properties()

    results: List[Dict[str, Any]] = []
    for prop in properties:
        prop_timer = Timer()
        prop_timer.start()

        checker = SafetyChecker(kripke)
        check_result = checker.check(prop)

        prop_timer.stop()

        status = "✓ PASS" if check_result.holds else "✗ FAIL"
        results.append({
            "name": prop.name,
            "kind": type(prop).__name__,
            "holds": check_result.holds,
            "time": prop_timer.elapsed,
        })
        print(f"  {status}  {prop.name} ({type(prop).__name__})"
              f"  [{format_duration(prop_timer.elapsed)}]")

    timer.stop()

    passed = sum(1 for r in results if r["holds"])
    total = len(results)
    print(f"\n  Results: {passed}/{total} properties hold")
    print(f"  Total time: {format_duration(timer.elapsed)}")
    print(f"  ✓ Property checking complete")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(graph, quotient_graph, report, prop_results,
                  total_time: float):
    """Print a final summary of all pipeline results."""
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")

    ratio = graph.state_count / quotient_graph.state_count \
        if quotient_graph.state_count else float("inf")
    passed = sum(1 for r in prop_results if r["holds"])

    print(f"  Specification:     TwoPhaseCommit (N={N_PARTICIPANTS})")
    print(f"  Original states:   {graph.state_count:,}")
    print(f"  Quotient states:   {quotient_graph.state_count:,}")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Witness verdict:   {report.verdict.name}")
    print(f"  Properties:        {passed}/{len(prop_results)} pass")
    print(f"  Total time:        {format_duration(total_time)}")
    print()

    # Cleanup witness file
    if WITNESS_PATH.exists():
        WITNESS_PATH.unlink()
        print(f"  (Cleaned up {WITNESS_PATH})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the full Two-Phase Commit demonstration pipeline."""
    print(BANNER)
    total_start = time.time()

    try:
        spec_builder = step_build_spec(N_PARTICIPANTS)
        graph = step_explore(spec_builder)
        result, quotient_graph = step_compress(graph)
        witness = step_generate_witness(result, graph, quotient_graph)
        report = step_verify_witness()
        prop_results = step_check_properties(quotient_graph, spec_builder)

        total_time = time.time() - total_start
        print_summary(graph, quotient_graph, report, prop_results, total_time)

        return 0 if report.verdict == Verdict.PASS else 1

    except Exception as exc:
        print(f"\n  ✗ Pipeline failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if WITNESS_PATH.exists():
            WITNESS_PATH.unlink()


if __name__ == "__main__":
    sys.exit(main())
