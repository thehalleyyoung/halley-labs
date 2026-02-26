#!/usr/bin/env python3
"""
Benchmark — CoaCert-TLA Demonstration
=======================================

Runs all built-in specifications through the full CoaCert pipeline,
collects metrics, and produces a comparison table plus LaTeX output.

Specs benchmarked:
  - TwoPhaseCommit (N=3)
  - LeaderElection (N=3)
  - Peterson (N=2)
  - Paxos (N_acceptors=3, N_values=2, max_ballot=2)

Usage:
    python -m examples.benchmark_demo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from coacert.specs import (
        SpecRegistry,
        TwoPhaseCommitSpec,
        LeaderElectionSpec,
        PetersonSpec,
        PaxosSpec,
    )
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
    from coacert.properties import SafetyChecker, KripkeAdapter
    from coacert.evaluation import (
        Timer,
        MetricsCollector,
        format_duration,
        BenchmarkRunner,
        BenchmarkConfig,
    )
except ImportError as exc:
    print(f"ERROR: Could not import coacert: {exc}")
    print("Run from the implementation/ directory or install coacert.")
    sys.exit(1)


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   CoaCert-TLA: Benchmark Suite Demonstration               ║
╚══════════════════════════════════════════════════════════════╝
"""

# Benchmark configurations: (name, spec_class, kwargs)
BENCHMARKS = [
    ("TwoPhaseCommit-3", TwoPhaseCommitSpec, {"n_participants": 3}),
    ("LeaderElection-3", LeaderElectionSpec, {"n_nodes": 3}),
    ("Peterson-2",       PetersonSpec,       {"n_processes": 2}),
    ("Paxos-3-2-2",      PaxosSpec,
     {"n_acceptors": 3, "n_values": 2, "max_ballot": 2}),
]


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_single_benchmark(name: str, spec_cls, kwargs: Dict[str, Any]
                         ) -> Dict[str, Any]:
    """Run one spec through the full pipeline and collect metrics."""
    witness_path = Path(f"bench_{name.lower()}.cwit")
    metrics: Dict[str, Any] = {"name": name, "status": "running"}

    try:
        timer_total = Timer()
        timer_total.start()

        # Build spec
        spec = spec_cls(**kwargs)
        module = spec.get_spec()
        config = spec.get_config()
        properties = spec.get_properties()

        # Explore
        from coacert.semantics import ActionEvaluator, Environment
        env = Environment()
        env.bind_constants(config["constants"])
        engine = ActionEvaluator(module, env)

        timer = Timer()
        timer.start()
        explorer = ExplicitStateExplorer(engine, depth_limit=100)
        graph = explorer.explore()
        stats = explorer.stats
        timer.stop()
        metrics["explore_time"] = timer.elapsed
        metrics["states"] = stats.states_explored
        metrics["transitions"] = stats.transitions_found

        # Compress
        timer = Timer()
        timer.start()
        ref = RefinementEngine(graph, strategy=RefinementStrategy.EAGER)
        result = ref.run()
        quotient = QuotientBuilder(result.partition, graph).build()
        timer.stop()
        metrics["compress_time"] = timer.elapsed
        metrics["quotient_states"] = quotient.state_count
        metrics["quotient_transitions"] = quotient.transition_count
        metrics["rounds"] = result.rounds
        metrics["ratio"] = (graph.state_count / quotient.state_count
                            if quotient.state_count else float("inf"))

        # Witness
        timer = Timer()
        timer.start()

        binding = EquivalenceBinding()
        for sid in graph.state_ids:
            binding.bind(sid, f"c_{result.partition.get_class(sid)}")

        ws = WitnessSet()
        for edge in quotient.edges:
            ws.add_transition(TransitionWitness(
                source_class=f"c_{edge.source}",
                target_class=f"c_{edge.target}",
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
        nbytes = wf.serialize(witness_path)
        timer.stop()
        metrics["witness_time"] = timer.elapsed
        metrics["witness_size"] = nbytes
        metrics["witness_count"] = ws.total_count

        # Verify
        timer = Timer()
        timer.start()
        report = verify_witness(str(witness_path))
        timer.stop()
        metrics["verify_time"] = timer.elapsed
        metrics["verdict"] = report.verdict.name

        # Properties
        timer = Timer()
        timer.start()
        kripke = KripkeAdapter(quotient)
        passed = 0
        total_props = len(properties)
        for prop in properties:
            checker = SafetyChecker(kripke)
            cr = checker.check(prop)
            if cr.holds:
                passed += 1
        timer.stop()
        metrics["property_time"] = timer.elapsed
        metrics["properties_passed"] = passed
        metrics["properties_total"] = total_props

        timer_total.stop()
        metrics["total_time"] = timer_total.elapsed
        metrics["status"] = "completed"

    except Exception as exc:
        metrics["status"] = "failed"
        metrics["error"] = str(exc)

    finally:
        if witness_path.exists():
            witness_path.unlink()

    return metrics


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def print_results_table(results: List[Dict[str, Any]]):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("  Benchmark Results")
    print(f"{'='*80}")

    # Header
    h = (f"  {'Spec':<20} {'States':>8} {'Quot':>6} {'Ratio':>7} "
         f"{'Verdict':>7} {'Props':>6} {'Time':>10}")
    print(h)
    print(f"  {'─'*20} {'─'*8} {'─'*6} {'─'*7} {'─'*7} {'─'*6} {'─'*10}")

    for r in results:
        if r["status"] == "completed":
            print(f"  {r['name']:<20} "
                  f"{r['states']:>8,} "
                  f"{r['quotient_states']:>6,} "
                  f"{r['ratio']:>6.1f}x "
                  f"{r['verdict']:>7} "
                  f"{r['properties_passed']}/{r['properties_total']:>3} "
                  f"{format_duration(r['total_time']):>10}")
        else:
            print(f"  {r['name']:<20} {'FAILED':>8}  "
                  f"{r.get('error', 'unknown')[:40]}")

    # Phase timing breakdown
    print(f"\n{'='*80}")
    print("  Phase Timing Breakdown")
    print(f"{'='*80}")
    h2 = (f"  {'Spec':<20} {'Explore':>9} {'Compress':>9} "
           f"{'Witness':>9} {'Verify':>9} {'Props':>9}")
    print(h2)
    print(f"  {'─'*20} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")

    for r in results:
        if r["status"] != "completed":
            continue
        print(f"  {r['name']:<20} "
              f"{format_duration(r['explore_time']):>9} "
              f"{format_duration(r['compress_time']):>9} "
              f"{format_duration(r['witness_time']):>9} "
              f"{format_duration(r['verify_time']):>9} "
              f"{format_duration(r['property_time']):>9}")

    # Witness stats
    print(f"\n{'='*80}")
    print("  Witness Statistics")
    print(f"{'='*80}")
    h3 = (f"  {'Spec':<20} {'Entries':>8} {'Size':>10} "
           f"{'B/state':>8} {'Rounds':>7}")
    print(h3)
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8} {'─'*7}")

    for r in results:
        if r["status"] != "completed":
            continue
        bps = (r["witness_size"] / r["states"]
               if r["states"] else 0)
        print(f"  {r['name']:<20} "
              f"{r['witness_count']:>8,} "
              f"{r['witness_size']:>9,}B "
              f"{bps:>7.1f} "
              f"{r['rounds']:>7}")


def print_latex_table(results: List[Dict[str, Any]]):
    """Generate LaTeX-formatted results table."""
    print(f"\n{'='*80}")
    print("  LaTeX Table")
    print(f"{'='*80}")
    print()
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{CoaCert-TLA Benchmark Results}")
    print(r"\label{tab:benchmark}")
    print(r"\begin{tabular}{lrrrrrr}")
    print(r"\toprule")
    print(r"Specification & $|S|$ & $|S/{\sim}|$ & Ratio "
          r"& Witness & Props & Time (s) \\")
    print(r"\midrule")

    for r in results:
        if r["status"] != "completed":
            continue
        name_tex = r["name"].replace("_", r"\_")
        print(f"  {name_tex} & "
              f"{r['states']:,} & "
              f"{r['quotient_states']:,} & "
              f"{r['ratio']:.1f}$\\times$ & "
              f"{r['verdict']} & "
              f"{r['properties_passed']}/{r['properties_total']} & "
              f"{r['total_time']:.2f} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print(BANNER)
    results: List[Dict[str, Any]] = []

    print(f"  Running {len(BENCHMARKS)} benchmarks...\n")

    for i, (name, spec_cls, kwargs) in enumerate(BENCHMARKS, 1):
        print(f"  [{i}/{len(BENCHMARKS)}] {name}...", end="", flush=True)
        t0 = time.time()
        r = run_single_benchmark(name, spec_cls, kwargs)
        elapsed = time.time() - t0
        status = "✓" if r["status"] == "completed" else "✗"
        print(f" {status} ({format_duration(elapsed)})")
        results.append(r)

    print_results_table(results)
    print_latex_table(results)

    # Overall summary
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] != "completed"]
    total_states = sum(r["states"] for r in completed)
    total_quot = sum(r["quotient_states"] for r in completed)
    total_time = sum(r["total_time"] for r in completed)

    print(f"{'─'*60}")
    print(f"  Overall: {len(completed)} completed, {len(failed)} failed")
    print(f"  Total states explored:  {total_states:,}")
    print(f"  Total quotient states:  {total_quot:,}")
    if total_quot:
        print(f"  Aggregate compression:  {total_states/total_quot:.1f}x")
    print(f"  Total time:             {format_duration(total_time)}")
    print()

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
