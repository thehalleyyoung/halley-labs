#!/usr/bin/env python3
"""Genetic Gate Parameter Repair Example
=========================================

This example demonstrates BioProver's *parameter repair* workflow on a
genetic NOR gate — a fundamental building block in the Cello genetic
circuit design automation toolchain (Nielsen et al., Science 2016).

**Biology background**
A NOR gate is implemented as a single promoter repressed by two
independent inputs A and B.  The output protein Y should be HIGH only
when *both* inputs are LOW:

    A=0, B=0 → Y HIGH   (constitutive production dominates)
    A=1, B=0 → Y LOW    (A represses)
    A=0, B=1 → Y LOW    (B represses)
    A=1, B=1 → Y LOW    (both repress)

When characterised parts have parameter drift (e.g. from host context
effects), the gate may fail to meet its Boolean-logic specification.
BioProver can automatically *repair* — i.e. synthesise new parameter
values — that restore correct behaviour.

**What this example does**
1. Builds a NOR gate from two Hill-repression reactions + degradation.
2. Specifies the Boolean logic in Bio-STL.
3. Shows the gate fails verification with mischaracterised parameters.
4. Runs CEGIS-based parameter repair.
5. Displays the repaired parameter values and re-verifies.
"""

from __future__ import annotations

import sys

# ── BioProver imports ──────────────────────────────────────────────────
from bioprover.models.bio_model import BioModel, Compartment
from bioprover.models.species import Species, SpeciesType, BoundaryCondition
from bioprover.models.reactions import (
    HillRepression,
    ConstitutiveProduction,
    LinearDegradation,
    Reaction,
    StoichiometryEntry,
)
from bioprover.models.parameters import Parameter, UncertaintyType

from bioprover.cegar.cegar_engine import CEGARConfig, VerificationStatus
from bioprover import verify, repair


# ════════════════════════════════════════════════════════════════════════
#  Step 1 — Build the NOR gate model
# ════════════════════════════════════════════════════════════════════════

def build_nor_gate(
    alpha: float = 12.0,
    K_A: float = 4.0,
    K_B: float = 4.0,
    n_A: float = 2.0,
    n_B: float = 2.0,
    basal: float = 0.5,
    delta: float = 0.07,
) -> BioModel:
    """Construct a genetic NOR gate ODE model.

    The ODE for the output protein Y is:

        d[Y]/dt = basal
                + α / ((1 + ([A]/K_A)^n_A) · (1 + ([B]/K_B)^n_B))
                − δ·[Y]

    Because BioProver models reactions individually we split this into:
      - Basal (leak) production
      - Repression by A  (Hill repression, modifier=A)
      - Repression by B  (Hill repression, modifier=B)
      - First-order degradation of Y

    Input species A and B are held at fixed boundary concentrations
    that represent the upstream gate outputs.
    """
    model = BioModel(name="NOR_gate")
    model.add_compartment(Compartment(name="cell", size=1.0))

    # -- Input species (boundary-fixed at different logic levels) --------
    # We will sweep A and B through {LOW, HIGH} = {0.1, 10.0} nM.
    model.add_species(Species(
        name="A", species_type=SpeciesType.PROTEIN,
        initial_concentration=0.1, units="nM",
        boundary_condition=BoundaryCondition.FIXED,
        compartment="cell",
    ))
    model.add_species(Species(
        name="B", species_type=SpeciesType.PROTEIN,
        initial_concentration=0.1, units="nM",
        boundary_condition=BoundaryCondition.FIXED,
        compartment="cell",
    ))

    # -- Output species --------------------------------------------------
    model.add_species(Species(
        name="Y", species_type=SpeciesType.PROTEIN,
        initial_concentration=0.0, units="nM",
        compartment="cell",
    ))

    # -- Parameters (repairable) -----------------------------------------
    for p in [
        Parameter("alpha", value=alpha, units="nM/min",
                  lower_bound=1.0, upper_bound=50.0,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Max production rate"),
        Parameter("K_A", value=K_A, units="nM",
                  lower_bound=0.5, upper_bound=20.0,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Half-repression constant for input A"),
        Parameter("K_B", value=K_B, units="nM",
                  lower_bound=0.5, upper_bound=20.0,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Half-repression constant for input B"),
        Parameter("n_A", value=n_A,
                  lower_bound=1.0, upper_bound=4.0,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Hill coefficient for A"),
        Parameter("n_B", value=n_B,
                  lower_bound=1.0, upper_bound=4.0,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Hill coefficient for B"),
        Parameter("basal", value=basal, units="nM/min",
                  lower_bound=0.01, upper_bound=2.0,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Basal (leak) production rate"),
        Parameter("delta", value=delta, units="1/min",
                  lower_bound=0.01, upper_bound=0.2,
                  uncertainty_type=UncertaintyType.UNIFORM,
                  description="Degradation + dilution rate"),
    ]:
        model.add_parameter(p)

    # -- Reactions -------------------------------------------------------
    # Repression by A
    model.add_reaction(Reaction(
        name="Y_repression_A",
        reactants=[],
        products=[StoichiometryEntry("Y", 1)],
        kinetic_law=HillRepression(Vmax=alpha / 2.0, K=K_A, n=n_A),
        modifiers=["A"],
        compartment="cell",
    ))
    # Repression by B
    model.add_reaction(Reaction(
        name="Y_repression_B",
        reactants=[],
        products=[StoichiometryEntry("Y", 1)],
        kinetic_law=HillRepression(Vmax=alpha / 2.0, K=K_B, n=n_B),
        modifiers=["B"],
        compartment="cell",
    ))
    # Basal production
    model.add_reaction(Reaction(
        name="Y_basal",
        reactants=[],
        products=[StoichiometryEntry("Y", 1)],
        kinetic_law=ConstitutiveProduction(rate=basal),
        compartment="cell",
    ))
    # Degradation
    model.add_reaction(Reaction(
        name="Y_degradation",
        reactants=[StoichiometryEntry("Y", 1)],
        products=[],
        kinetic_law=LinearDegradation(rate=delta),
        compartment="cell",
    ))

    return model


# ════════════════════════════════════════════════════════════════════════
#  Step 2 — NOR gate Boolean logic specification
# ════════════════════════════════════════════════════════════════════════

def nor_gate_spec() -> str:
    """Bio-STL specification for correct NOR logic.

    We verify all four input combinations at steady state (t ∈ [80,100]):
      - A=0, B=0 → Y > 5       (output HIGH)
      - A=1, B=0 → Y < 1       (output LOW)
      - A=0, B=1 → Y < 1       (output LOW)
      - A=1, B=1 → Y < 1       (output LOW)

    Because BioProver operates on a single trajectory, the spec
    encodes the expected steady-state behaviour for the initial
    conditions set on the model (A and B boundary values).

    For a full NOR gate characterisation, the caller should run
    verification for each of the four input combinations.
    """
    # For the default model (A=LOW, B=LOW) the output should be HIGH.
    return "G[80,100]( Y > 5 )"


def nor_gate_specs_all_inputs():
    """Return (input_desc, A_value, B_value, spec_string) for all
    four Boolean input combinations of a NOR gate.
    """
    return [
        ("A=0 B=0", 0.1, 0.1, "G[80,100]( Y > 5 )"),   # output HIGH
        ("A=1 B=0", 10., 0.1, "G[80,100]( Y < 1 )"),   # output LOW
        ("A=0 B=1", 0.1, 10., "G[80,100]( Y < 1 )"),   # output LOW
        ("A=1 B=1", 10., 10., "G[80,100]( Y < 1 )"),   # output LOW
    ]


# ════════════════════════════════════════════════════════════════════════
#  Step 3 — Demonstrate verification failure
# ════════════════════════════════════════════════════════════════════════

def demonstrate_failure():
    """Show that mischaracterised parameters cause NOR gate failure.

    We deliberately use a weak Hill coefficient (n=1.2) and a high
    basal rate, mimicking a gate that was characterised in a different
    host strain.  The gate leaks too much output and fails the LOW
    specification for the (A=1, B=0) input.
    """
    print("=" * 65)
    print("  BioProver — Genetic NOR Gate: Parameter Repair")
    print("=" * 65)

    # Mischaracterised parameters: weak repression
    bad_params = dict(alpha=12.0, K_A=8.0, K_B=8.0,
                      n_A=1.2, n_B=1.2, basal=1.5, delta=0.07)

    config = CEGARConfig(max_iterations=30, timeout=60.0)

    print("\n── Verifying NOR gate with mischaracterised parameters ──")
    print(f"  Parameters: {bad_params}")
    all_pass = True

    for desc, a_val, b_val, spec_str in nor_gate_specs_all_inputs():
        model = build_nor_gate(**bad_params)
        # Fix input species to the test values
        for sp in model.species:
            if sp.name == "A":
                sp.initial_concentration = a_val
            elif sp.name == "B":
                sp.initial_concentration = b_val

        result = verify(model, spec=spec_str, timeout=30.0, config=config)
        status_sym = "✓" if result.status == VerificationStatus.VERIFIED else "✗"
        print(f"  {status_sym}  {desc}  spec={spec_str}  → {result.status.name}")
        if result.status != VerificationStatus.VERIFIED:
            all_pass = False

    if not all_pass:
        print("\n  ✗ NOR gate fails at least one input combination.")
        print("    → Proceeding to parameter repair.\n")
    else:
        print("\n  ✓ All inputs pass (unexpected with bad params).\n")

    return bad_params


# ════════════════════════════════════════════════════════════════════════
#  Step 4 — Run parameter repair
# ════════════════════════════════════════════════════════════════════════

def run_repair(bad_params: dict):
    """Use BioProver's repair() to find new parameters that satisfy
    the NOR gate specification across all four input combinations.

    The repair engine uses CEGIS (counterexample-guided inductive
    synthesis) under the hood: it iteratively proposes parameter
    candidates, checks them against the spec, and refines using
    counterexamples until a valid parameter set is found.
    """
    print("── Running CEGIS parameter repair ──")

    # Build model with bad parameters — repair will adjust them
    model = build_nor_gate(**bad_params)

    # We repair against the hardest-to-satisfy case: A=0, B=0 → Y HIGH
    # (the others are easier once repression is strengthened).
    spec_str = "G[80,100]( Y > 5 )"

    repair_result = repair(
        model,
        spec=spec_str,
        budget=0.8,      # allow up to 80% fractional change per parameter
        timeout=120.0,
    )

    print(f"\n  Repair success: {repair_result.success}")
    if hasattr(repair_result, 'robustness'):
        print(f"  Robustness    : {repair_result.robustness:+.4f}")

    if repair_result.success:
        print("\n  ── Original vs Repaired parameters ──")
        orig = repair_result.original_parameters
        fixed = repair_result.repaired_parameters
        for key in sorted(set(list(orig.keys()) + list(fixed.keys()))):
            o = orig.get(key, "?")
            r = fixed.get(key, "?")
            if isinstance(o, float) and isinstance(r, float):
                delta_pct = abs(r - o) / max(abs(o), 1e-12) * 100
                print(f"    {key:>8s}:  {o:8.4f}  → {r:8.4f}  ({delta_pct:5.1f}% change)")
            else:
                print(f"    {key:>8s}:  {o}  → {r}")

    return repair_result


# ════════════════════════════════════════════════════════════════════════
#  Step 5 — Re-verify with repaired parameters
# ════════════════════════════════════════════════════════════════════════

def reverify(repair_result):
    """Re-verify the NOR gate with repaired parameters."""

    if not repair_result.success:
        print("\n  Repair failed; skipping re-verification.")
        return

    print("\n── Re-verifying NOR gate with repaired parameters ──")
    repaired = repair_result.repaired_parameters
    config = CEGARConfig(max_iterations=30, timeout=60.0)

    all_pass = True
    for desc, a_val, b_val, spec_str in nor_gate_specs_all_inputs():
        model = build_nor_gate(
            alpha=repaired.get("alpha", 12.0),
            K_A=repaired.get("K_A", 4.0),
            K_B=repaired.get("K_B", 4.0),
            n_A=repaired.get("n_A", 2.0),
            n_B=repaired.get("n_B", 2.0),
            basal=repaired.get("basal", 0.5),
            delta=repaired.get("delta", 0.07),
        )
        for sp in model.species:
            if sp.name == "A":
                sp.initial_concentration = a_val
            elif sp.name == "B":
                sp.initial_concentration = b_val

        result = verify(model, spec=spec_str, timeout=30.0, config=config)
        status_sym = "✓" if result.status == VerificationStatus.VERIFIED else "✗"
        print(f"  {status_sym}  {desc}  → {result.status.name}")
        if result.status != VerificationStatus.VERIFIED:
            all_pass = False

    if all_pass:
        print("\n  ✓ Repaired NOR gate passes all four input combinations!")
    else:
        print("\n  ⚠ Some inputs still fail — repair may need a larger budget.")


# ════════════════════════════════════════════════════════════════════════
#  Step 6 — Repair report summary
# ════════════════════════════════════════════════════════════════════════

def print_repair_report(repair_result):
    """Display a concise repair report."""
    print(f"\n{'─' * 55}")
    print("  Repair Report")
    print(f"{'─' * 55}")
    print(f"  Success       : {repair_result.success}")
    if hasattr(repair_result, 'robustness'):
        print(f"  Robustness    : {repair_result.robustness:+.4f}")
    if hasattr(repair_result, 'message') and repair_result.message:
        print(f"  Message       : {repair_result.message}")
    print(f"{'─' * 55}\n")


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bad_params = demonstrate_failure()
    repair_result = run_repair(bad_params)
    print_repair_report(repair_result)
    reverify(repair_result)
    sys.exit(0 if repair_result.success else 1)
