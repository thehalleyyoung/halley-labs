#!/usr/bin/env python3
"""Benchmark Comparison Example
================================

This example runs BioProver on a set of standard synthetic biology
benchmark circuits and compares the full system (with AI-guided
refinement) against the BioProver-no-AI ablation.

**Motivation**
A key claim of BioProver is that AI-guided predicate selection
accelerates CEGAR convergence.  This script provides a reproducible
comparison by running both configurations on the same benchmark suite
and displaying a results table with speedup metrics.

**Benchmarks included**
1. Toggle switch — bistability (2 species, EASY)
2. Repressilator — oscillation (3 species, MEDIUM)
3. NOR gate — Boolean logic (3 species, EASY)
4. 3-stage cascade — signal propagation (4 species, MEDIUM)
5. NAND-NOR circuit — 2-gate composition (5 species, HARD)

**What this example does**
1. Defines each benchmark circuit inline (model + Bio-STL spec).
2. Runs BioProver with AI heuristics enabled.
3. Runs BioProver with AI heuristics disabled (ablation baseline).
4. Tabulates results: status, time, CEGAR iterations, speedup.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

# ── BioProver imports ──────────────────────────────────────────────────
from bioprover.models.bio_model import BioModel, Compartment
from bioprover.models.species import Species, SpeciesType, BoundaryCondition
from bioprover.models.reactions import (
    HillRepression,
    HillActivation,
    LinearDegradation,
    ConstitutiveProduction,
    Reaction,
    StoichiometryEntry,
)
from bioprover.models.parameters import Parameter, UncertaintyType
from bioprover.cegar.cegar_engine import CEGARConfig, VerificationStatus
from bioprover import verify


# ════════════════════════════════════════════════════════════════════════
#  Data structures
# ════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkCase:
    """A single benchmark circuit with its specification."""
    name: str
    difficulty: str
    model: BioModel
    spec: str
    num_species: int = 0

    def __post_init__(self):
        self.num_species = len(self.model.species_names)


@dataclass
class BenchmarkRun:
    """Result of running a single benchmark."""
    name: str
    status: str
    time_s: float
    iterations: int
    coverage: float
    robustness: float
    ai_enabled: bool


# ════════════════════════════════════════════════════════════════════════
#  Benchmark circuit builders
# ════════════════════════════════════════════════════════════════════════

def _toggle_switch() -> BenchmarkCase:
    """Toggle switch bistability benchmark."""
    m = BioModel(name="toggle_switch")
    m.add_compartment(Compartment("cell"))
    for name, ic in [("LacI", 10.0), ("TetR", 0.5)]:
        m.add_species(Species(name, SpeciesType.PROTEIN, ic, "nM"))
    m.add_parameter(Parameter("alpha", 15.0, lower_bound=12.0, upper_bound=18.0,
                              uncertainty_type=UncertaintyType.UNIFORM))
    for prod, rep in [("LacI", "TetR"), ("TetR", "LacI")]:
        m.add_reaction(Reaction(f"{prod}_prod", [],
                                [StoichiometryEntry(prod, 1)],
                                HillRepression(Vmax=15.0, K=5.0, n=2.5),
                                modifiers=[rep]))
        m.add_reaction(Reaction(f"{prod}_deg",
                                [StoichiometryEntry(prod, 1)], [],
                                LinearDegradation(rate=0.069)))
    spec = "F[0,200]( G[0,100]( (LacI > 8 & TetR < 2) | (LacI < 2 & TetR > 8) ) )"
    return BenchmarkCase("toggle_switch", "EASY", m, spec)


def _repressilator() -> BenchmarkCase:
    """Repressilator oscillation benchmark."""
    m = BioModel(name="repressilator")
    m.add_compartment(Compartment("cell"))
    for name, ic in [("LacI", 8.0), ("TetR", 2.0), ("CI", 2.0)]:
        m.add_species(Species(name, SpeciesType.PROTEIN, ic, "nM"))
    m.add_parameter(Parameter("alpha", 16.0, lower_bound=13.0, upper_bound=19.0,
                              uncertainty_type=UncertaintyType.UNIFORM))
    ring = [("CI", "LacI"), ("LacI", "TetR"), ("TetR", "CI")]
    for rep, prod in ring:
        m.add_reaction(Reaction(f"{prod}_prod", [],
                                [StoichiometryEntry(prod, 1)],
                                HillRepression(Vmax=16.0, K=5.0, n=2.0),
                                modifiers=[rep]))
        m.add_reaction(Reaction(f"{prod}_deg",
                                [StoichiometryEntry(prod, 1)], [],
                                LinearDegradation(rate=0.069)))
    spec = "G[0,600]( F[0,300](LacI > 7) & F[0,300](LacI < 3) )"
    return BenchmarkCase("repressilator", "MEDIUM", m, spec)


def _nor_gate() -> BenchmarkCase:
    """NOR gate Boolean logic benchmark."""
    m = BioModel(name="nor_gate")
    m.add_compartment(Compartment("cell"))
    m.add_species(Species("A", SpeciesType.PROTEIN, 0.1, "nM",
                          boundary_condition=BoundaryCondition.FIXED))
    m.add_species(Species("B", SpeciesType.PROTEIN, 0.1, "nM",
                          boundary_condition=BoundaryCondition.FIXED))
    m.add_species(Species("Y", SpeciesType.PROTEIN, 0.0, "nM"))
    m.add_parameter(Parameter("alpha", 12.0, lower_bound=9.0, upper_bound=15.0,
                              uncertainty_type=UncertaintyType.UNIFORM))
    m.add_reaction(Reaction("Y_repr_A", [], [StoichiometryEntry("Y", 1)],
                            HillRepression(Vmax=6.0, K=4.0, n=2.0), modifiers=["A"]))
    m.add_reaction(Reaction("Y_repr_B", [], [StoichiometryEntry("Y", 1)],
                            HillRepression(Vmax=6.0, K=4.0, n=2.0), modifiers=["B"]))
    m.add_reaction(Reaction("Y_basal", [], [StoichiometryEntry("Y", 1)],
                            ConstitutiveProduction(rate=0.3)))
    m.add_reaction(Reaction("Y_deg", [StoichiometryEntry("Y", 1)], [],
                            LinearDegradation(rate=0.07)))
    spec = "G[80,100]( Y > 5 )"
    return BenchmarkCase("nor_gate", "EASY", m, spec)


def _cascade_3() -> BenchmarkCase:
    """3-stage activation cascade benchmark."""
    m = BioModel(name="cascade_3")
    m.add_compartment(Compartment("cell"))
    m.add_species(Species("Input", SpeciesType.SMALL_MOLECULE, 10.0, "nM",
                          boundary_condition=BoundaryCondition.FIXED))
    for name in ["A", "B", "Y"]:
        m.add_species(Species(name, SpeciesType.PROTEIN, 0.0, "nM"))
    m.add_parameter(Parameter("alpha", 15.0, lower_bound=12.0, upper_bound=18.0,
                              uncertainty_type=UncertaintyType.UNIFORM))
    for act, prod in [("Input", "A"), ("A", "B"), ("B", "Y")]:
        m.add_reaction(Reaction(f"{prod}_prod", [], [StoichiometryEntry(prod, 1)],
                                HillActivation(Vmax=15.0, K=5.0, n=2.0),
                                modifiers=[act]))
        m.add_reaction(Reaction(f"{prod}_deg", [StoichiometryEntry(prod, 1)], [],
                                LinearDegradation(rate=0.07)))
    spec = "G[80,100]( Y > 2 )"
    return BenchmarkCase("cascade_3", "MEDIUM", m, spec)


def _nand_nor() -> BenchmarkCase:
    """2-gate NAND-NOR circuit benchmark (harder)."""
    m = BioModel(name="nand_nor")
    m.add_compartment(Compartment("cell"))
    m.add_species(Species("A", SpeciesType.PROTEIN, 10.0, "nM",
                          boundary_condition=BoundaryCondition.FIXED))
    m.add_species(Species("B", SpeciesType.PROTEIN, 10.0, "nM",
                          boundary_condition=BoundaryCondition.FIXED))
    m.add_species(Species("G1", SpeciesType.PROTEIN, 0.0, "nM"))  # NAND output
    m.add_species(Species("C",  SpeciesType.PROTEIN, 0.1, "nM",
                          boundary_condition=BoundaryCondition.FIXED))
    m.add_species(Species("Y",  SpeciesType.PROTEIN, 0.0, "nM"))  # NOR output
    m.add_parameter(Parameter("alpha", 14.0, lower_bound=11.0, upper_bound=17.0,
                              uncertainty_type=UncertaintyType.UNIFORM))
    # NAND gate: G1 high unless A AND B both high
    m.add_reaction(Reaction("G1_repr_A", [], [StoichiometryEntry("G1", 1)],
                            HillRepression(Vmax=7.0, K=5.0, n=2.0), modifiers=["A"]))
    m.add_reaction(Reaction("G1_repr_B", [], [StoichiometryEntry("G1", 1)],
                            HillRepression(Vmax=7.0, K=5.0, n=2.0), modifiers=["B"]))
    m.add_reaction(Reaction("G1_basal", [], [StoichiometryEntry("G1", 1)],
                            ConstitutiveProduction(rate=0.5)))
    m.add_reaction(Reaction("G1_deg", [StoichiometryEntry("G1", 1)], [],
                            LinearDegradation(rate=0.07)))
    # NOR gate: Y high unless G1 OR C high
    m.add_reaction(Reaction("Y_repr_G1", [], [StoichiometryEntry("Y", 1)],
                            HillRepression(Vmax=7.0, K=5.0, n=2.0), modifiers=["G1"]))
    m.add_reaction(Reaction("Y_repr_C", [], [StoichiometryEntry("Y", 1)],
                            HillRepression(Vmax=7.0, K=5.0, n=2.0), modifiers=["C"]))
    m.add_reaction(Reaction("Y_basal", [], [StoichiometryEntry("Y", 1)],
                            ConstitutiveProduction(rate=0.3)))
    m.add_reaction(Reaction("Y_deg", [StoichiometryEntry("Y", 1)], [],
                            LinearDegradation(rate=0.07)))
    # With A=high, B=high → G1 low (NAND). C=low → Y = NOR(G1,C) = NOR(low,low) = high
    spec = "G[80,100]( Y > 3 )"
    return BenchmarkCase("nand_nor", "HARD", m, spec)


# ════════════════════════════════════════════════════════════════════════
#  Run benchmarks
# ════════════════════════════════════════════════════════════════════════

def run_benchmark(case: BenchmarkCase, ai_enabled: bool,
                  timeout: float = 60.0) -> BenchmarkRun:
    """Run a single benchmark with or without AI heuristics."""
    config = CEGARConfig(
        max_iterations=50,
        timeout=timeout,
        enable_ai_heuristic=ai_enabled,
        enable_bounded_guarantee=True,
    )
    t0 = time.time()
    result = verify(case.model, spec=case.spec, timeout=timeout, config=config)
    dt = time.time() - t0

    iters = result.statistics.iterations if result.statistics else 0
    return BenchmarkRun(
        name=case.name,
        status=result.status.name,
        time_s=dt,
        iterations=iters,
        coverage=result.coverage,
        robustness=result.robustness,
        ai_enabled=ai_enabled,
    )


def run_all():
    """Run all benchmarks with and without AI, display comparison."""

    print("=" * 72)
    print("  BioProver — Benchmark Comparison: Full vs No-AI Ablation")
    print("=" * 72)

    benchmarks = [
        _toggle_switch(),
        _repressilator(),
        _nor_gate(),
        _cascade_3(),
        _nand_nor(),
    ]

    results_ai: List[BenchmarkRun] = []
    results_no_ai: List[BenchmarkRun] = []

    for case in benchmarks:
        print(f"\n  Running: {case.name} ({case.difficulty}, "
              f"{case.num_species} species) ...")

        # With AI
        r_ai = run_benchmark(case, ai_enabled=True, timeout=60.0)
        results_ai.append(r_ai)

        # Without AI (ablation)
        r_no = run_benchmark(case, ai_enabled=False, timeout=60.0)
        results_no_ai.append(r_no)

        print(f"    AI:    {r_ai.status:18s}  {r_ai.time_s:6.2f}s  "
              f"{r_ai.iterations:3d} iters")
        print(f"    No-AI: {r_no.status:18s}  {r_no.time_s:6.2f}s  "
              f"{r_no.iterations:3d} iters")

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    header = (f"  {'Benchmark':<16s} {'Diff':>5s} │ "
              f"{'AI time':>8s} {'iters':>5s} │ "
              f"{'NoAI time':>9s} {'iters':>5s} │ {'Speedup':>7s}")
    print(header)
    print(f"  {'─' * 16} {'─' * 5} ┼ {'─' * 8} {'─' * 5} ┼ "
          f"{'─' * 9} {'─' * 5} ┼ {'─' * 7}")

    for ai, no in zip(results_ai, results_no_ai):
        speedup = no.time_s / max(ai.time_s, 0.01)
        case = next(b for b in benchmarks if b.name == ai.name)
        print(f"  {ai.name:<16s} {case.difficulty:>5s} │ "
              f"{ai.time_s:7.2f}s {ai.iterations:5d} │ "
              f"{no.time_s:8.2f}s {no.iterations:5d} │ "
              f"{speedup:6.1f}×")

    # Aggregate
    total_ai = sum(r.time_s for r in results_ai)
    total_no = sum(r.time_s for r in results_no_ai)
    print(f"  {'─' * 16} {'─' * 5} ┼ {'─' * 8} {'─' * 5} ┼ "
          f"{'─' * 9} {'─' * 5} ┼ {'─' * 7}")
    overall = total_no / max(total_ai, 0.01)
    print(f"  {'TOTAL':<16s} {'':>5s} │ "
          f"{total_ai:7.2f}s {'':>5s} │ "
          f"{total_no:8.2f}s {'':>5s} │ "
          f"{overall:6.1f}×")
    print(f"{'═' * 72}\n")


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_all()
