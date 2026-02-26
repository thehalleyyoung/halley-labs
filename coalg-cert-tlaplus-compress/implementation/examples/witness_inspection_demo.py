#!/usr/bin/env python3
"""
Witness Inspection — CoaCert-TLA Demonstration
================================================

Demonstrates how to inspect a generated bisimulation witness
certificate in detail:

  1. Generate a witness from a small spec
  2. Inspect its internal structure
  3. Verify individual Merkle proofs for equivalence classes
  4. Print detailed witness statistics

Usage:
    python -m examples.witness_inspection_demo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from coacert.specs import TwoPhaseCommitSpec
    from coacert.explorer import ExplicitStateExplorer, TransitionGraph
    from coacert.bisimulation import (
        RefinementEngine,
        RefinementStrategy,
        QuotientBuilder,
    )
    from coacert.witness import (
        WitnessSet,
        TransitionWitness,
        StutterWitness,
        FairnessWitness,
        EquivalenceBinding,
        HashChain,
        WitnessFormat,
        MerkleTree,
        MerkleProof,
    )
    from coacert.verifier import (
        verify_witness,
        WitnessDeserializer,
        HashChainVerifier,
        ClosureValidator,
        Verdict,
    )
    from coacert.evaluation import Timer, format_duration
except ImportError as exc:
    print(f"ERROR: Could not import coacert: {exc}")
    print("Run from the implementation/ directory or install coacert.")
    sys.exit(1)


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   CoaCert-TLA: Witness Inspection Demonstration            ║
╚══════════════════════════════════════════════════════════════╝
"""

WITNESS_PATH = Path("inspection_demo_witness.cwit")


# ---------------------------------------------------------------------------
# Generate a witness to inspect
# ---------------------------------------------------------------------------

def generate_witness() -> Dict[str, Any]:
    """Run the pipeline on 2PC (N=3) to produce a witness file."""
    print("\n  Generating witness from TwoPhaseCommit (N=3)...")
    spec = TwoPhaseCommitSpec(n_participants=3)
    module = spec.get_spec()
    config = spec.get_config()

    from coacert.semantics import ActionEvaluator, Environment
    env = Environment()
    env.bind_constants(config["constants"])
    engine = ActionEvaluator(module, env)

    explorer = ExplicitStateExplorer(engine, depth_limit=100)
    graph = explorer.explore()

    ref = RefinementEngine(graph, strategy=RefinementStrategy.EAGER)
    result = ref.run()
    quotient = QuotientBuilder(result.partition, graph).build()

    # Build witness
    binding = EquivalenceBinding()
    class_map: Dict[Any, str] = {}
    for sid in graph.state_ids:
        cid = result.partition.get_class(sid)
        cls_name = f"class_{cid}"
        binding.bind(sid, cls_name)
        class_map[sid] = cls_name

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

    wf = WitnessFormat(
        equivalence=binding, witnesses=ws, chain=chain,
        original_state_count=graph.state_count,
        quotient_state_count=quotient.state_count,
    )
    nbytes = wf.serialize(WITNESS_PATH)

    print(f"  ✓ Witness generated: {WITNESS_PATH} ({nbytes:,} bytes)")
    return {
        "graph": graph,
        "quotient": quotient,
        "partition": result.partition,
        "binding": binding,
        "witness_set": ws,
        "chain": chain,
        "format": wf,
        "class_map": class_map,
    }


# ---------------------------------------------------------------------------
# Inspect witness structure
# ---------------------------------------------------------------------------

def inspect_structure(data: Dict[str, Any]):
    """Print detailed information about witness internals."""
    print(f"\n{'='*60}")
    print("  Witness Structure")
    print(f"{'='*60}")

    ws: WitnessSet = data["witness_set"]
    binding: EquivalenceBinding = data["binding"]
    chain: HashChain = data["chain"]
    wf: WitnessFormat = data["format"]

    # Equivalence binding stats
    print(f"\n  Equivalence Binding:")
    print(f"    Total states bound: {binding.state_count}")
    print(f"    Equivalence classes: {binding.class_count}")
    classes = binding.get_all_classes()
    sizes = [len(members) for members in classes.values()]
    if sizes:
        print(f"    Min class size:    {min(sizes)}")
        print(f"    Max class size:    {max(sizes)}")
        print(f"    Avg class size:    {sum(sizes)/len(sizes):.1f}")

    # Show a few equivalence classes
    print(f"\n    Sample classes (first 5):")
    for i, (cls_name, members) in enumerate(list(classes.items())[:5]):
        member_str = ", ".join(str(m) for m in list(members)[:3])
        suffix = f"... (+{len(members)-3})" if len(members) > 3 else ""
        print(f"      {cls_name}: [{member_str}{suffix}]"
              f" ({len(members)} states)")

    # Transition witnesses
    print(f"\n  Transition Witnesses:")
    print(f"    Count:   {ws.transition_count}")
    transitions = ws.transitions
    if transitions:
        print(f"    Sample (first 5):")
        for tw in transitions[:5]:
            print(f"      {tw.source_class} --{tw.action!r}--> "
                  f"{tw.target_class}")
            print(f"        digest: {tw.digest.hex()[:32]}...")

    # Stutter witnesses
    print(f"\n  Stutter Witnesses:  {ws.stutter_count}")
    stutters = ws.stutters
    if stutters:
        for sw in stutters[:3]:
            print(f"    class={sw.equiv_class}, "
                  f"path_len={len(sw.path)}")

    # Fairness witnesses
    print(f"\n  Fairness Witnesses: {ws.fairness_count}")

    # Hash chain
    print(f"\n  Hash Chain:")
    print(f"    Blocks:     {chain.block_count}")
    print(f"    Chain hash: {chain.digest.hex()[:32]}...")

    # Overall Merkle tree
    print(f"\n  Merkle Tree:")
    print(f"    Root:       {ws.root.hex()[:32]}...")
    print(f"    Leaf count: {ws.total_count}")


# ---------------------------------------------------------------------------
# Verify individual Merkle proofs
# ---------------------------------------------------------------------------

def verify_merkle_proofs(data: Dict[str, Any]):
    """Verify Merkle proofs for individual witness entries."""
    print(f"\n{'='*60}")
    print("  Merkle Proof Verification")
    print(f"{'='*60}")

    ws: WitnessSet = data["witness_set"]
    transitions = ws.transitions

    if not transitions:
        print("  No transitions to verify.")
        return

    # Build a Merkle tree from all witness digests
    all_digests = [tw.digest for tw in transitions]
    all_digests.extend(sw.digest for sw in ws.stutters)
    all_digests.extend(fw.digest for fw in ws.fairness_witnesses)

    tree = MerkleTree(all_digests)
    root = tree.root

    print(f"  Tree root: {root.hex()[:32]}...")
    print(f"  Total leaves: {len(all_digests)}")

    # Verify proofs for the first few transitions
    n_to_check = min(5, len(transitions))
    print(f"\n  Verifying {n_to_check} proofs:")
    all_ok = True
    for i in range(n_to_check):
        proof = tree.proof(i)
        valid = proof.verify(all_digests[i], root)
        status = "✓" if valid else "✗"
        if not valid:
            all_ok = False
        print(f"    [{i}] {status}  leaf={all_digests[i].hex()[:16]}... "
              f"path_len={len(proof.siblings)}")

    if all_ok:
        print(f"\n  ✓ All {n_to_check} Merkle proofs verified successfully")
    else:
        print(f"\n  ✗ Some Merkle proofs failed")


# ---------------------------------------------------------------------------
# Verify via verifier module
# ---------------------------------------------------------------------------

def verify_full_witness():
    """Run the full verifier pipeline on the serialized witness."""
    print(f"\n{'='*60}")
    print("  Full Witness Verification")
    print(f"{'='*60}")

    timer = Timer()
    timer.start()
    report = verify_witness(str(WITNESS_PATH))
    timer.stop()

    print(f"  Verdict: {report.verdict.name}")
    if report.errors:
        print(f"  Errors ({len(report.errors)}):")
        for err in report.errors[:10]:
            print(f"    - {err}")
    else:
        print(f"  No errors found")
    print(f"  Time: {format_duration(timer.elapsed)}")

    return report


# ---------------------------------------------------------------------------
# Print witness file statistics
# ---------------------------------------------------------------------------

def print_file_stats(data: Dict[str, Any]):
    """Print statistics about the witness file."""
    print(f"\n{'='*60}")
    print("  Witness File Statistics")
    print(f"{'='*60}")

    wf: WitnessFormat = data["format"]
    graph: TransitionGraph = data["graph"]
    quotient: TransitionGraph = data["quotient"]

    file_size = WITNESS_PATH.stat().st_size if WITNESS_PATH.exists() else 0
    breakdown = wf.size_breakdown()

    print(f"  File path:              {WITNESS_PATH}")
    print(f"  Total size:             {file_size:,} bytes")
    print(f"  Original states:        {graph.state_count:,}")
    print(f"  Quotient states:        {quotient.state_count:,}")
    ratio = graph.state_count / quotient.state_count \
        if quotient.state_count else 0
    print(f"  Compression ratio:      {ratio:.2f}x")
    print(f"  Bytes per orig state:   "
          f"{file_size / graph.state_count:.1f}" if graph.state_count else "N/A")
    print(f"  Bytes per quot state:   "
          f"{file_size / quotient.state_count:.1f}" if quotient.state_count else "N/A")

    if breakdown:
        print(f"\n  Size Breakdown:")
        for section, size in breakdown.items():
            pct = 100.0 * size / file_size if file_size else 0
            print(f"    {section:<20} {size:>8,} bytes ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(BANNER)

    try:
        data = generate_witness()
        inspect_structure(data)
        verify_merkle_proofs(data)
        report = verify_full_witness()
        print_file_stats(data)

        print(f"\n{'─'*60}")
        print(f"  Witness inspection complete.")
        print(f"  Final verdict: {report.verdict.name}")
        print(f"{'─'*60}\n")

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
