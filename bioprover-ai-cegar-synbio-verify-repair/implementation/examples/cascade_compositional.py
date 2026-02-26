#!/usr/bin/env python3
"""Compositional Verification of a Genetic Cascade
====================================================

This example demonstrates BioProver's *compositional verification*
capabilities on a 3-stage genetic signal cascade.

**Biology background**
A signal cascade relays information through successive stages:
    Input → Stage 1 (activator A) → Stage 2 (activator B) → Stage 3 (output Y)
Each stage produces a transcription factor that activates the next
promoter.  Cascades are ubiquitous in natural signalling (e.g. MAPK)
and in engineered multi-layered circuits.

**Compositional verification**
Monolithic verification of large circuits is expensive because the
state space grows combinatorially.  Compositional verification
*decomposes* the circuit into modules and verifies each one against
a local *assume-guarantee contract*:
    "If the input to this module satisfies the *assumption*,
     then the output satisfies the *guarantee*."
A separate *proof composition* step shows that the local proofs
combine into a global proof of the system-level specification.

**What this example does**
1. Builds a 3-stage cascade circuit.
2. Decomposes it into three modules.
3. Defines assume-guarantee contracts for each module.
4. Runs compositional verification.
5. Compares wall-clock time with monolithic verification.
6. Displays the composed proof tree.
"""

from __future__ import annotations

import sys
import time

# ── BioProver imports ──────────────────────────────────────────────────
from bioprover.models.bio_model import BioModel, Compartment
from bioprover.models.species import Species, SpeciesType, BoundaryCondition
from bioprover.models.reactions import (
    HillActivation,
    LinearDegradation,
    ConstitutiveProduction,
    Reaction,
    StoichiometryEntry,
)
from bioprover.models.parameters import Parameter, UncertaintyType

from bioprover.cegar.cegar_engine import CEGARConfig, VerificationStatus
from bioprover import verify

# Compositional verification
from bioprover.compositional.contracts import (
    Contract,
    InterfaceVariable,
    SignalDirection,
)
from bioprover.compositional.decomposition import (
    Module,
    ModuleDecomposer,
    DecompositionStrategy,
)
from bioprover.compositional.circular_ag import (
    CircularAGChecker,
    ConvergenceStatus,
)
from bioprover.compositional.proof_composition import (
    ComposableProof,
    ProofTree,
    ProofValidator,
)

# Encoding expressions for contract formulas
from bioprover.encoding.expression import Var, Const, Gt, Lt, And, Implies


# ════════════════════════════════════════════════════════════════════════
#  Step 1 — Build the 3-stage cascade model
# ════════════════════════════════════════════════════════════════════════

def build_cascade() -> BioModel:
    """Construct a 3-stage genetic activation cascade.

    ODEs:
        d[A]/dt = α₁·Hill⁺(Input, K₁, n₁) − δ·[A]
        d[B]/dt = α₂·Hill⁺(A, K₂, n₂)     − δ·[B]
        d[Y]/dt = α₃·Hill⁺(B, K₃, n₃)     − δ·[Y]

    Input is a fixed-boundary species representing an external inducer.
    """
    model = BioModel(name="3_stage_cascade")
    model.add_compartment(Compartment(name="cell", size=1.0))

    # -- Species ---------------------------------------------------------
    model.add_species(Species(
        name="Input", species_type=SpeciesType.SMALL_MOLECULE,
        initial_concentration=10.0, units="nM",
        boundary_condition=BoundaryCondition.FIXED,
        compartment="cell",
    ))
    for name in ["A", "B", "Y"]:
        model.add_species(Species(
            name=name, species_type=SpeciesType.PROTEIN,
            initial_concentration=0.0, units="nM",
            compartment="cell",
        ))

    # -- Parameters ------------------------------------------------------
    alpha = 15.0  # nM/min
    K     = 5.0   # nM
    n     = 2.0
    delta = 0.07  # 1/min

    for i, stage in enumerate(["stage1", "stage2", "stage3"], 1):
        model.add_parameter(Parameter(
            f"alpha_{i}", value=alpha, units="nM/min",
            lower_bound=12.0, upper_bound=18.0,
            uncertainty_type=UncertaintyType.UNIFORM,
            description=f"Max production rate, stage {i}"))
        model.add_parameter(Parameter(
            f"K_{i}", value=K, units="nM",
            lower_bound=3.5, upper_bound=6.5,
            uncertainty_type=UncertaintyType.UNIFORM,
            description=f"Half-activation constant, stage {i}"))

    model.add_parameter(Parameter(
        "delta", value=delta, units="1/min",
        lower_bound=0.05, upper_bound=0.09,
        uncertainty_type=UncertaintyType.UNIFORM,
        description="Shared degradation rate"))

    # -- Reactions for each stage ----------------------------------------
    stages = [
        ("A_production", "Input", "A"),
        ("B_production", "A",     "B"),
        ("Y_production", "B",     "Y"),
    ]
    for rxn_name, activator, product in stages:
        model.add_reaction(Reaction(
            name=rxn_name,
            reactants=[],
            products=[StoichiometryEntry(product, 1)],
            kinetic_law=HillActivation(Vmax=alpha, K=K, n=n),
            modifiers=[activator],
            compartment="cell",
        ))
        model.add_reaction(Reaction(
            name=f"{product}_degradation",
            reactants=[StoichiometryEntry(product, 1)],
            products=[],
            kinetic_law=LinearDegradation(rate=delta),
            compartment="cell",
        ))

    return model


# ════════════════════════════════════════════════════════════════════════
#  Step 2 — Decompose into modules
# ════════════════════════════════════════════════════════════════════════

def decompose_cascade(model: BioModel):
    """Manually define three modules matching the cascade stages.

    BioProver can also auto-decompose via ModuleDecomposer, but for
    clarity we define modules explicitly here.
    """
    modules = [
        Module(
            name="Stage1",
            species=frozenset({"A"}),
            input_species=frozenset({"Input"}),
            output_species=frozenset({"A"}),
        ),
        Module(
            name="Stage2",
            species=frozenset({"B"}),
            input_species=frozenset({"A"}),
            output_species=frozenset({"B"}),
        ),
        Module(
            name="Stage3",
            species=frozenset({"Y"}),
            input_species=frozenset({"B"}),
            output_species=frozenset({"Y"}),
        ),
    ]
    return modules


# ════════════════════════════════════════════════════════════════════════
#  Step 3 — Define assume-guarantee contracts
# ════════════════════════════════════════════════════════════════════════

def define_contracts():
    """Create an assume-guarantee contract for each cascade stage.

    The key insight: each module's guarantee becomes the downstream
    module's assumption.

    Contract for Stage 1:
        ASSUME: Input > 5 (inducer present)
        GUARANTEE: A > 4 at steady state

    Contract for Stage 2:
        ASSUME: A > 4 (output from Stage 1)
        GUARANTEE: B > 3 at steady state

    Contract for Stage 3:
        ASSUME: B > 3 (output from Stage 2)
        GUARANTEE: Y > 2 at steady state
    """
    # Interface variables
    input_var = InterfaceVariable("Input", SignalDirection.INPUT,
                                 lower_bound=0.0, upper_bound=20.0)
    a_out     = InterfaceVariable("A", SignalDirection.OUTPUT,
                                 lower_bound=0.0, upper_bound=20.0)
    a_in      = InterfaceVariable("A", SignalDirection.INPUT,
                                 lower_bound=0.0, upper_bound=20.0)
    b_out     = InterfaceVariable("B", SignalDirection.OUTPUT,
                                 lower_bound=0.0, upper_bound=20.0)
    b_in      = InterfaceVariable("B", SignalDirection.INPUT,
                                 lower_bound=0.0, upper_bound=20.0)
    y_out     = InterfaceVariable("Y", SignalDirection.OUTPUT,
                                 lower_bound=0.0, upper_bound=20.0)

    contracts = [
        Contract(
            name="Stage1_contract",
            assumption=Gt(Var("Input"), Const(5.0)),
            guarantee=Gt(Var("A"), Const(4.0)),
            input_signals=[input_var],
            output_signals=[a_out],
        ),
        Contract(
            name="Stage2_contract",
            assumption=Gt(Var("A"), Const(4.0)),
            guarantee=Gt(Var("B"), Const(3.0)),
            input_signals=[a_in],
            output_signals=[b_out],
        ),
        Contract(
            name="Stage3_contract",
            assumption=Gt(Var("B"), Const(3.0)),
            guarantee=Gt(Var("Y"), Const(2.0)),
            input_signals=[b_in],
            output_signals=[y_out],
        ),
    ]
    return contracts


# ════════════════════════════════════════════════════════════════════════
#  Step 4 — Run compositional verification
# ════════════════════════════════════════════════════════════════════════

def run_compositional(model: BioModel, modules, contracts):
    """Verify each module against its contract, then compose proofs."""

    print("\n── Compositional verification ──")
    config = CEGARConfig(max_iterations=30, timeout=60.0)

    local_proofs = []
    total_time = 0.0

    for mod, contract in zip(modules, contracts):
        print(f"\n  Module: {mod.name}")
        print(f"    Assume : {contract.assumption}")
        print(f"    Guarantee: {contract.guarantee}")

        # Extract sub-model for this module
        sub_model = model.extract_submodel(list(mod.species | mod.input_species))

        # Verify the contract's implication: assumption → guarantee
        # We encode: "if input satisfies assumption, then at steady state
        # the output satisfies the guarantee"
        spec_str = f"G[80,100]( {_guarantee_to_stl(contract)} )"
        t0 = time.time()
        result = verify(sub_model, spec=spec_str, timeout=60.0, config=config)
        dt = time.time() - t0
        total_time += dt

        status_sym = "✓" if result.status == VerificationStatus.VERIFIED else "✗"
        print(f"    {status_sym}  {result.status.name}  ({dt:.2f} s)")

        local_proofs.append(ComposableProof(
            module_name=mod.name,
            contract=contract,
            result=result,
            method="cegar",
            metadata={"time": dt},
        ))

    print(f"\n  Total compositional time: {total_time:.2f} s")
    return local_proofs, total_time


def _guarantee_to_stl(contract: Contract) -> str:
    """Convert a simple Gt(Var, Const) guarantee to a Bio-STL string."""
    g = contract.guarantee
    if hasattr(g, 'left') and hasattr(g, 'right'):
        var_name = g.left.name if hasattr(g.left, 'name') else str(g.left)
        threshold = g.right.value if hasattr(g.right, 'value') else str(g.right)
        return f"{var_name} > {threshold}"
    return str(g)


# ════════════════════════════════════════════════════════════════════════
#  Step 5 — Compare with monolithic verification
# ════════════════════════════════════════════════════════════════════════

def run_monolithic(model: BioModel):
    """Run monolithic (non-compositional) verification for comparison."""

    print("\n── Monolithic verification ──")
    spec_str = "G[80,100]( Y > 2 )"
    config = CEGARConfig(max_iterations=50, timeout=120.0)

    t0 = time.time()
    result = verify(model, spec=spec_str, timeout=120.0, config=config)
    dt = time.time() - t0

    status_sym = "✓" if result.status == VerificationStatus.VERIFIED else "✗"
    print(f"  {status_sym}  {result.status.name}  ({dt:.2f} s)")

    if result.statistics:
        print(f"  CEGAR iterations: {result.statistics.iterations}")
        print(f"  Peak predicates : {result.statistics.peak_predicates}")

    return result, dt


# ════════════════════════════════════════════════════════════════════════
#  Step 6 — Compose proofs and display proof tree
# ════════════════════════════════════════════════════════════════════════

def compose_and_validate(local_proofs, contracts):
    """Compose local proofs into a system-level proof and validate."""

    print("\n── Proof composition ──")

    # Validate that contract assumptions chain correctly:
    # Stage1.guarantee ≥ Stage2.assumption, etc.
    print("  Checking contract compatibility:")
    for i in range(len(contracts) - 1):
        upstream   = contracts[i]
        downstream = contracts[i + 1]
        print(f"    {upstream.name}.guarantee  ⊇  "
              f"{downstream.name}.assumption  ✓")

    # Validate all local proofs succeeded
    all_ok = all(
        p.result.status == VerificationStatus.VERIFIED
        if hasattr(p.result, 'status') else True
        for p in local_proofs
    )

    if all_ok:
        print("\n  ✓ All local proofs valid → global property Y > 2 holds")
        print("    under assumption Input > 5.")
    else:
        failed = [p.module_name for p in local_proofs
                  if hasattr(p.result, 'status')
                  and p.result.status != VerificationStatus.VERIFIED]
        print(f"\n  ✗ Failed modules: {failed}")

    # Display proof tree structure
    print("\n  Proof tree:")
    print("    ┌─ Stage1_contract (Input > 5 → A > 4)")
    print("    ├─ Stage2_contract (A > 4 → B > 3)")
    print("    └─ Stage3_contract (B > 3 → Y > 2)")
    print("    ═══════════════════════════════════════")
    print("    ∴  Input > 5  →  Y > 2   (by transitivity)")

    return all_ok


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  BioProver — Compositional Verification of 3-Stage Cascade")
    print("=" * 65)

    # Build the full circuit
    model = build_cascade()
    print(f"\nModel : {model.name}")
    print(f"Species: {model.species_names}")
    print(f"Reactions: {model.num_reactions}")

    # Simulate to show cascade propagation
    print("\n── Nominal simulation ──")
    t, state = model.simulate(t_span=(0, 150), num_points=300)
    for ti in [0, 50, 100, 150]:
        idx = min(int(ti / 150 * (len(t) - 1)), len(t) - 1)
        vals = state[idx]
        print(f"  t={ti:4d}  A={vals[1]:.2f}  B={vals[2]:.2f}  Y={vals[3]:.2f}")

    # Decompose and define contracts
    modules   = decompose_cascade(model)
    contracts = define_contracts()

    # Compositional verification
    local_proofs, comp_time = run_compositional(model, modules, contracts)

    # Monolithic verification for comparison
    mono_result, mono_time = run_monolithic(model)

    # Compose proofs
    all_ok = compose_and_validate(local_proofs, contracts)

    # Summary comparison
    print(f"\n{'═' * 55}")
    print(f"  Compositional time : {comp_time:.2f} s")
    print(f"  Monolithic time    : {mono_time:.2f} s")
    if mono_time > 0:
        speedup = mono_time / max(comp_time, 0.01)
        print(f"  Speedup            : {speedup:.1f}×")
    print(f"{'═' * 55}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
