#!/usr/bin/env python3
"""Toggle Switch Verification Example
======================================

This example demonstrates BioProver's core verification workflow on a
genetic toggle switch — the canonical bistable synthetic circuit first
constructed by Gardner, Cantor & Collins (Nature, 2000).

**Biology background**
A toggle switch consists of two mutually repressing transcription factors:
  - LacI represses TetR production
  - TetR represses LacI production
Under the right parameter regime the circuit exhibits *bistability*:
it settles into one of two stable steady states (LacI-high/TetR-low
or LacI-low/TetR-high) depending on initial conditions and inducers.

**What this example does**
1. Builds the toggle switch ODE model from first principles.
2. Attaches literature-sourced parameters with realistic uncertainty.
3. Encodes the bistability specification in Bio-STL.
4. Runs CEGAR-based verification.
5. Inspects the result: proof certificate or counterexample trace.
"""

from __future__ import annotations

import sys
import math

# ── BioProver core imports ──────────────────────────────────────────────
from bioprover.models.bio_model import BioModel, Compartment
from bioprover.models.species import Species, SpeciesType
from bioprover.models.reactions import (
    HillRepression,
    LinearDegradation,
    ConstitutiveProduction,
    Reaction,
    StoichiometryEntry,
)
from bioprover.models.parameters import Parameter, ParameterSet, UncertaintyType

# Temporal logic (Bio-STL) AST nodes
from bioprover.temporal.stl_ast import (
    Predicate,
    Expression,
    Always,
    Eventually,
    STLAnd,
    STLOr,
    STLNot,
    Interval,
    ComparisonOp,
)

# Verification engine
from bioprover.cegar.cegar_engine import CEGARConfig, CEGAREngine, VerificationStatus

# Top-level convenience function (parses Bio-STL string → runs CEGAR)
from bioprover import verify


# ════════════════════════════════════════════════════════════════════════
#  Step 1 — Build the toggle switch model
# ════════════════════════════════════════════════════════════════════════

def build_toggle_switch() -> BioModel:
    """Construct the Gardner toggle switch ODE model.

    The ODEs are:
        d[LacI]/dt = α₁ / (1 + ([TetR]/K₁)^n₁) − δ₁·[LacI]
        d[TetR]/dt = α₂ / (1 + ([LacI]/K₂)^n₂) − δ₂·[TetR]

    where αᵢ is the maximal production rate, Kᵢ is the half-repression
    constant, nᵢ is the Hill coefficient, and δᵢ is the degradation
    rate.
    """
    model = BioModel(name="toggle_switch")

    # -- Compartment (single well-mixed cell) ----------------------------
    model.add_compartment(Compartment(name="cell", size=1.0, units="litre"))

    # -- Species ---------------------------------------------------------
    # LacI starts high, TetR starts low → one of the two stable states.
    laci = Species(
        name="LacI",
        species_type=SpeciesType.PROTEIN,
        initial_concentration=10.0,  # nM
        units="nM",
        compartment="cell",
    )
    tetr = Species(
        name="TetR",
        species_type=SpeciesType.PROTEIN,
        initial_concentration=0.5,   # nM
        units="nM",
        compartment="cell",
    )
    model.add_species(laci)
    model.add_species(tetr)

    # -- Parameters with uncertainty -------------------------------------
    # Literature values from Gardner et al. (2000) & Lugagne et al. (2017)
    # with ±20 % uniform uncertainty to reflect strain-to-strain variation.

    params = [
        # Maximal production rates (nM/min)
        Parameter("alpha1", value=15.6, units="nM/min",
                  lower_bound=12.5, upper_bound=18.7,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Max production rate of LacI promoter"),
        Parameter("alpha2", value=13.0, units="nM/min",
                  lower_bound=10.4, upper_bound=15.6,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Max production rate of TetR promoter"),
        # Half-repression constants (nM)
        Parameter("K1", value=5.0, units="nM",
                  lower_bound=4.0, upper_bound=6.0,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Half-repression constant for TetR → LacI"),
        Parameter("K2", value=5.0, units="nM",
                  lower_bound=4.0, upper_bound=6.0,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Half-repression constant for LacI → TetR"),
        # Hill coefficients (dimensionless) — treated as fixed
        Parameter("n1", value=2.5, uncertainty_type=UncertaintyType.FIXED,
                  description="Hill coefficient for TetR repression of LacI"),
        Parameter("n2", value=2.5, uncertainty_type=UncertaintyType.FIXED,
                  description="Hill coefficient for LacI repression of TetR"),
        # Degradation/dilution rates (1/min)
        Parameter("delta1", value=0.069, units="1/min",
                  lower_bound=0.055, upper_bound=0.083,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="LacI degradation + dilution rate"),
        Parameter("delta2", value=0.069, units="1/min",
                  lower_bound=0.055, upper_bound=0.083,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="TetR degradation + dilution rate"),
    ]
    for p in params:
        model.add_parameter(p)

    # -- Reactions -------------------------------------------------------
    # 1) LacI production: repressed by TetR  (Hill repression)
    model.add_reaction(Reaction(
        name="LacI_production",
        reactants=[],
        products=[StoichiometryEntry(species_name="LacI", coefficient=1)],
        kinetic_law=HillRepression(Vmax=15.6, K=5.0, n=2.5),
        modifiers=["TetR"],       # TetR is the repressor input
        compartment="cell",
    ))

    # 2) TetR production: repressed by LacI  (Hill repression)
    model.add_reaction(Reaction(
        name="TetR_production",
        reactants=[],
        products=[StoichiometryEntry(species_name="TetR", coefficient=1)],
        kinetic_law=HillRepression(Vmax=13.0, K=5.0, n=2.5),
        modifiers=["LacI"],       # LacI is the repressor input
        compartment="cell",
    ))

    # 3) LacI degradation (first-order)
    model.add_reaction(Reaction(
        name="LacI_degradation",
        reactants=[StoichiometryEntry(species_name="LacI", coefficient=1)],
        products=[],
        kinetic_law=LinearDegradation(rate=0.069),
        compartment="cell",
    ))

    # 4) TetR degradation (first-order)
    model.add_reaction(Reaction(
        name="TetR_degradation",
        reactants=[StoichiometryEntry(species_name="TetR", coefficient=1)],
        products=[],
        kinetic_law=LinearDegradation(rate=0.069),
        compartment="cell",
    ))

    return model


# ════════════════════════════════════════════════════════════════════════
#  Step 2 — Define the bistability specification in Bio-STL
# ════════════════════════════════════════════════════════════════════════

def build_bistability_spec_ast():
    """Bistability specification as an STL formula AST.

    We require that the system eventually settles into *one* of two
    separated steady states and then *stays* there.

    Formally (Bio-STL):
        ◇[0,200] □[0,100] (
            (LacI > 8  ∧ TetR < 2)     ← "LacI-high" state
          ∨ (LacI < 2  ∧ TetR > 8)     ← "TetR-high" state
        )

    In words: within 200 minutes, the system reaches and then maintains
    for 100 minutes one of the two bistable attractors.
    """

    # Atomic predicates
    laci_high = Predicate(Expression(variable="LacI"), ComparisonOp.GT, 8.0)
    tetr_low  = Predicate(Expression(variable="TetR"), ComparisonOp.LT, 2.0)
    laci_low  = Predicate(Expression(variable="LacI"), ComparisonOp.LT, 2.0)
    tetr_high = Predicate(Expression(variable="TetR"), ComparisonOp.GT, 8.0)

    # State A: LacI-high / TetR-low
    state_a = STLAnd(laci_high, tetr_low)
    # State B: LacI-low / TetR-high
    state_b = STLAnd(laci_low, tetr_high)

    # Must be in one of the two states
    in_bistable_attractor = STLOr(state_a, state_b)

    # Must stay there for 100 minutes
    persist = Always(in_bistable_attractor, Interval(0.0, 100.0))

    # Must reach within 200 minutes
    spec = Eventually(persist, Interval(0.0, 200.0))

    return spec


def bistability_spec_string() -> str:
    """Same specification expressed as a Bio-STL string.

    The ``verify()`` convenience function can parse this directly.
    """
    return (
        "F[0,200]( G[0,100]( "
        "  (LacI > 8 & TetR < 2) | (LacI < 2 & TetR > 8) "
        ") )"
    )


# ════════════════════════════════════════════════════════════════════════
#  Step 3 — Run verification
# ════════════════════════════════════════════════════════════════════════

def run_verification():
    """Execute CEGAR verification of the toggle switch bistability."""

    print("=" * 65)
    print("  BioProver — Toggle Switch Bistability Verification")
    print("=" * 65)

    # Build model
    model = build_toggle_switch()
    print(f"\nModel : {model.name}")
    print(f"Species: {model.species_names}")
    print(f"Reactions: {model.num_reactions}")

    # (Optional) quick ODE simulation to visualise trajectory
    print("\n── Simulating nominal trajectory ──")
    t, state = model.simulate(t_span=(0, 300), num_points=500)
    final_laci = state[-1][0]
    final_tetr = state[-1][1]
    print(f"  t=0   : LacI={state[0][0]:.2f} nM, TetR={state[0][1]:.2f} nM")
    print(f"  t=300 : LacI={final_laci:.2f} nM, TetR={final_tetr:.2f} nM")

    # ── Method A: convenience verify() with Bio-STL string ──────────
    print("\n── Verification (Bio-STL string API) ──")
    spec_str = bistability_spec_string()
    print(f"  Spec: {spec_str}")

    config = CEGARConfig(
        max_iterations=50,
        timeout=120.0,           # 2-minute timeout for this example
        enable_ai_heuristic=False,
        enable_bounded_guarantee=True,
    )

    result = verify(
        model,
        spec=spec_str,
        mode="full",
        timeout=120.0,
        config=config,
    )

    # ── Display results ─────────────────────────────────────────────
    print_result(result)

    return result


# ════════════════════════════════════════════════════════════════════════
#  Step 4 — Inspect results
# ════════════════════════════════════════════════════════════════════════

def print_result(result):
    """Pretty-print a VerificationResult."""

    print(f"\n{'─' * 50}")
    print(f"  Status  : {result.status.name}")
    print(f"  Coverage: {result.coverage * 100:.1f} %")
    print(f"  Robustness: {result.robustness:+.4f}")

    if result.statistics is not None:
        stats = result.statistics
        print(f"  CEGAR iterations : {stats.iterations}")
        print(f"  Wall-clock time  : {stats.total_time:.2f} s")
        print(f"  Peak predicates  : {stats.peak_predicates}")
        print(f"  Spurious CEXs    : {stats.spurious_count}")

    # ── Counterexample trace ────────────────────────────────────────
    if result.status == VerificationStatus.FALSIFIED and result.counterexample:
        cex = result.counterexample
        print("\n  ── Counterexample trace ──")
        print(f"  Property violated: {cex.property_violated}")
        print(f"  Parameters: {cex.parameter_values}")
        n_pts = len(cex.time_points)
        # Show first, middle, and last time points
        for idx in [0, n_pts // 2, n_pts - 1]:
            t = cex.time_points[idx]
            s = cex.states[idx]
            print(f"    t={t:7.2f}  LacI={s.get('LacI', 0):.3f}  "
                  f"TetR={s.get('TetR', 0):.3f}")

    elif result.status == VerificationStatus.VERIFIED:
        print("\n  ✓ Bistability verified for all parameters in the")
        print("    uncertainty envelope.")
        if result.proof_certificate:
            print(f"  Proof certificate keys: "
                  f"{list(result.proof_certificate.keys())}")

    elif result.status == VerificationStatus.BOUNDED_GUARANTEE:
        print(f"\n  ⊞ Bounded guarantee: property holds over "
              f"{result.coverage * 100:.1f}% of parameter space.")

    print(f"{'─' * 50}\n")


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    result = run_verification()
    sys.exit(0 if result.status != VerificationStatus.FALSIFIED else 1)
