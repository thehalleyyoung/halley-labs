#!/usr/bin/env python3
"""
Custom Specification — CoaCert-TLA Demonstration
==================================================

Shows how to build custom TLA-lite specifications programmatically
using the ``ModuleBuilder`` and AST utilities, then runs each through
the full CoaCert pipeline.

Two custom specs are demonstrated:

  1. **Counter Mod N** — a single counter that increments modulo N.
     Trivial but illustrates the API and achieves high compression.

  2. **Producer-Consumer** — a bounded buffer with a producer and
     consumer process.  Demonstrates multi-variable specs, guards,
     and safety properties.

Usage:
    python -m examples.custom_spec_demo
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from coacert.specs.spec_utils import (
        ModuleBuilder,
        ident,
        primed,
        int_lit,
        bool_lit,
        str_lit,
        make_conjunction,
        make_disjunction,
        make_eq,
        make_neq,
        make_lt,
        make_plus,
        make_minus,
        make_mod,
        make_primed_eq,
        make_guard,
        make_unchanged,
        make_function_construction,
        make_func_apply,
        make_set_enum,
        make_int_range,
        make_string_set,
        make_forall_single,
        make_exists_single,
        make_in,
        make_invariant_property,
        make_safety_property,
        make_liveness_property,
        make_always,
        make_eventually,
        make_vars_tuple,
        make_spec_with_fairness,
        make_wf,
        make_land,
        make_lor,
        make_geq,
        make_implies,
    )
    from coacert.parser import PrettyPrinter
    from coacert.parser.ast_nodes import Module, Property
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
    from coacert.evaluation import Timer, format_duration
except ImportError as exc:
    print(f"ERROR: Could not import coacert: {exc}")
    print("Run from the implementation/ directory or install coacert.")
    sys.exit(1)


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   CoaCert-TLA: Custom Specification Demonstration          ║
╚══════════════════════════════════════════════════════════════╝
"""


# ===================================================================
# Custom Spec 1: Counter Mod N
# ===================================================================

def build_counter_spec(n: int) -> Tuple[Module, List[Property], Dict[str, Any]]:
    """Build a simple counter that increments modulo N.

    Variables: counter
    Init:     counter = 0
    Next:     counter' = (counter + 1) % N
    Property: counter is always in 0..N-1
    """
    mb = ModuleBuilder("CounterModN")
    mb.add_extends("Naturals")
    mb.add_constants("N")
    mb.add_variables("counter")

    # Init: counter = 0
    init = make_eq(ident("counter"), int_lit(0))
    mb.add_definition("Init", init)

    # Next: counter' = (counter + 1) % N
    next_expr = make_primed_eq(
        "counter",
        make_mod(make_plus(ident("counter"), int_lit(1)), ident("N")),
    )
    mb.add_definition("Next", next_expr)

    # TypeOK: counter \in 0..(N-1)
    type_ok = make_in(
        ident("counter"),
        make_int_range(int_lit(0), make_minus(ident("N"), int_lit(1))),
    )
    mb.add_definition("TypeOK", type_ok)

    # Recurrence: counter eventually equals 0
    from coacert.parser.ast_nodes import EventuallyExpr, AlwaysExpr
    recurrence = AlwaysExpr(
        expr=EventuallyExpr(
            expr=make_eq(ident("counter"), int_lit(0)),
        ),
    )
    mb.add_definition("Recurrence", recurrence)

    # Spec with fairness
    vars_t = make_vars_tuple("counter")
    fairness = [make_wf(vars_t, ident("Next"))]
    spec_expr = make_spec_with_fairness("Init", "Next", vars_t, fairness)
    mb.add_definition("Spec", spec_expr)

    # Properties
    props = [
        make_invariant_property("TypeOK", ident("TypeOK")),
        make_liveness_property("Recurrence", ident("Recurrence")),
    ]
    for p in props:
        mb.add_property(p)

    config = {"constants": {"N": n}, "spec_name": "CounterModN"}
    return mb.build(), props, config


# ===================================================================
# Custom Spec 2: Producer-Consumer with bounded buffer
# ===================================================================

def build_producer_consumer_spec(buf_size: int) -> Tuple[Module, List[Property], Dict[str, Any]]:
    """Build a producer-consumer with a bounded buffer.

    Variables: buf_count, prod_state, cons_state
    The producer adds to the buffer when not full,
    the consumer removes when not empty.
    """
    mb = ModuleBuilder("ProducerConsumer")
    mb.add_extends("Naturals")
    mb.add_constants("BufSize")
    mb.add_variables("buf_count", "prod_state", "cons_state")

    # Init
    init = make_conjunction([
        make_eq(ident("buf_count"), int_lit(0)),
        make_eq(ident("prod_state"), str_lit("idle")),
        make_eq(ident("cons_state"), str_lit("idle")),
    ])
    mb.add_definition("Init", init)

    # Produce: buf_count < BufSize => buf_count' = buf_count + 1
    produce = make_guard(
        make_land(
            make_eq(ident("prod_state"), str_lit("idle")),
            make_lt(ident("buf_count"), ident("BufSize")),
        ),
        [
            make_primed_eq("buf_count",
                           make_plus(ident("buf_count"), int_lit(1))),
            make_primed_eq("prod_state", str_lit("producing")),
        ],
        ["cons_state"],
    )
    mb.add_definition("Produce", produce)

    # ProduceDone: prod goes back to idle
    produce_done = make_guard(
        make_eq(ident("prod_state"), str_lit("producing")),
        [make_primed_eq("prod_state", str_lit("idle"))],
        ["buf_count", "cons_state"],
    )
    mb.add_definition("ProduceDone", produce_done)

    # Consume: buf_count > 0 => buf_count' = buf_count - 1
    consume = make_guard(
        make_land(
            make_eq(ident("cons_state"), str_lit("idle")),
            make_geq(ident("buf_count"), int_lit(1)),
        ),
        [
            make_primed_eq("buf_count",
                           make_minus(ident("buf_count"), int_lit(1))),
            make_primed_eq("cons_state", str_lit("consuming")),
        ],
        ["prod_state"],
    )
    mb.add_definition("Consume", consume)

    # ConsumeDone
    consume_done = make_guard(
        make_eq(ident("cons_state"), str_lit("consuming")),
        [make_primed_eq("cons_state", str_lit("idle"))],
        ["buf_count", "prod_state"],
    )
    mb.add_definition("ConsumeDone", consume_done)

    # Next
    next_expr = make_disjunction([
        ident("Produce"),
        ident("ProduceDone"),
        ident("Consume"),
        ident("ConsumeDone"),
    ])
    mb.add_definition("Next", next_expr)

    # TypeOK
    type_ok = make_conjunction([
        make_in(ident("buf_count"),
                make_int_range(int_lit(0), ident("BufSize"))),
        make_in(ident("prod_state"),
                make_string_set("idle", "producing")),
        make_in(ident("cons_state"),
                make_string_set("idle", "consuming")),
    ])
    mb.add_definition("TypeOK", type_ok)

    # NoOverflow: buf_count <= BufSize
    from coacert.specs.spec_utils import make_leq
    no_overflow = make_leq(ident("buf_count"), ident("BufSize"))
    mb.add_definition("NoOverflow", no_overflow)

    # NoUnderflow: buf_count >= 0
    no_underflow = make_geq(ident("buf_count"), int_lit(0))
    mb.add_definition("NoUnderflow", no_underflow)

    # Spec
    vars_t = make_vars_tuple("buf_count", "prod_state", "cons_state")
    fairness = [
        make_wf(vars_t, ident("Produce")),
        make_wf(vars_t, ident("Consume")),
        make_wf(vars_t, ident("ProduceDone")),
        make_wf(vars_t, ident("ConsumeDone")),
    ]
    spec_expr = make_spec_with_fairness("Init", "Next", vars_t, fairness)
    mb.add_definition("Spec", spec_expr)

    props = [
        make_invariant_property("TypeOK", ident("TypeOK")),
        make_safety_property("NoOverflow", ident("NoOverflow")),
        make_safety_property("NoUnderflow", ident("NoUnderflow")),
    ]
    for p in props:
        mb.add_property(p)

    config = {"constants": {"BufSize": buf_size},
              "spec_name": "ProducerConsumer"}
    return mb.build(), props, config


# ===================================================================
# Pipeline runner
# ===================================================================

def run_pipeline(name: str, module: Module, properties: List[Property],
                 config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full CoaCert pipeline on a custom spec."""
    print(f"\n{'─'*60}")
    print(f"  Pipeline: {name}")
    print(f"{'─'*60}")
    witness_path = Path(f"{name.lower().replace(' ', '_')}_witness.cwit")
    total_start = time.time()

    # Print spec info
    pp = PrettyPrinter()
    source = pp.print_module(module)
    lines = source.strip().split("\n")
    print(f"  Source: {len(lines)} lines")
    for line in lines[:6]:
        print(f"    {line}")
    if len(lines) > 6:
        print(f"    ... ({len(lines) - 6} more)")

    # Explore
    from coacert.semantics import ActionEvaluator, Environment
    env = Environment()
    env.bind_constants(config["constants"])
    engine = ActionEvaluator(module, env)

    timer = Timer()
    timer.start()
    explorer = ExplicitStateExplorer(engine, depth_limit=50)
    graph = explorer.explore()
    stats = explorer.stats
    timer.stop()
    explore_time = timer.elapsed

    print(f"\n  Exploration: {stats.states_explored:,} states, "
          f"{stats.transitions_found:,} transitions "
          f"[{format_duration(explore_time)}]")

    # Compress
    timer = Timer()
    timer.start()
    ref_engine = RefinementEngine(graph, strategy=RefinementStrategy.EAGER)
    result = ref_engine.run()
    quotient = QuotientBuilder(result.partition, graph).build()
    timer.stop()
    compress_time = timer.elapsed

    ratio = graph.state_count / quotient.state_count \
        if quotient.state_count else float("inf")
    print(f"  Compression: {graph.state_count:,} → {quotient.state_count:,} "
          f"({ratio:.1f}x) [{format_duration(compress_time)}]")

    # Witness
    timer = Timer()
    timer.start()

    binding = EquivalenceBinding()
    for sid in graph.state_ids:
        binding.bind(sid, f"class_{result.partition.get_class(sid)}")

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
    nbytes = wf.serialize(witness_path)
    report = verify_witness(str(witness_path))
    timer.stop()
    witness_time = timer.elapsed

    print(f"  Witness:     {ws.total_count} entries, {nbytes:,} bytes, "
          f"{report.verdict.name} [{format_duration(witness_time)}]")

    # Properties
    timer = Timer()
    timer.start()
    kripke = KripkeAdapter(quotient)
    prop_results = []
    for prop in properties:
        checker = SafetyChecker(kripke)
        cr = checker.check(prop)
        mark = "✓" if cr.holds else "✗"
        prop_results.append({"name": prop.name, "holds": cr.holds})
        print(f"  Property:    {mark} {prop.name}")
    timer.stop()
    prop_time = timer.elapsed

    total = time.time() - total_start
    passed = sum(1 for r in prop_results if r["holds"])

    print(f"\n  Result: {passed}/{len(prop_results)} properties pass, "
          f"total {format_duration(total)}")

    # Cleanup
    if witness_path.exists():
        witness_path.unlink()

    return {
        "name": name,
        "states": graph.state_count,
        "quotient": quotient.state_count,
        "ratio": ratio,
        "verdict": report.verdict.name,
        "properties": f"{passed}/{len(prop_results)}",
        "time": total,
    }


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    print(BANNER)
    all_results: List[Dict[str, Any]] = []

    try:
        # Spec 1: Counter Mod 8
        module1, props1, config1 = build_counter_spec(8)
        r1 = run_pipeline("CounterMod8", module1, props1, config1)
        all_results.append(r1)

        # Spec 2: Producer-Consumer (buf=3)
        module2, props2, config2 = build_producer_consumer_spec(3)
        r2 = run_pipeline("ProducerConsumer", module2, props2, config2)
        all_results.append(r2)

        # Summary table
        print(f"\n{'='*60}")
        print("  Custom Spec Results Summary")
        print(f"{'='*60}")
        print(f"  {'Spec':<22} {'States':>8} {'Quot':>6} "
              f"{'Ratio':>7} {'Verdict':>8} {'Props':>6}")
        print(f"  {'─'*22} {'─'*8} {'─'*6} {'─'*7} {'─'*8} {'─'*6}")
        for r in all_results:
            print(f"  {r['name']:<22} {r['states']:>8,} {r['quotient']:>6,} "
                  f"{r['ratio']:>6.1f}x {r['verdict']:>8} {r['properties']:>6}")
        print()

        return 0

    except Exception as exc:
        print(f"\n  ✗ Failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
