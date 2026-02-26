#!/usr/bin/env python3
"""Stochastic Circuit Analysis Example
========================================

This example demonstrates BioProver's stochastic analysis tools on a
toggle switch operating in the *low copy-number* regime, where
deterministic ODE models break down and stochastic effects dominate.

**Biology background**
At low molecule counts (< ~100 copies), the discrete, random nature
of biochemical reactions matters.  A toggle switch that is deterministically
bistable may exhibit *stochastic switching* — rare noise-driven
transitions between stable states.  Predicting the switching probability
is critical for circuit reliability.

**Analysis methods demonstrated**
1. **Stochastic Simulation Algorithm (SSA)** — exact Gillespie simulation
   of individual trajectories.
2. **Ensemble statistics** — mean, variance, and percentiles from many
   SSA trajectories.
3. **Moment closure** — approximate the probability distribution by
   tracking mean and (co)variance via moment ODEs with a normal closure.
4. **Hybrid SSA/ODE** — partition species into stochastic (low-count)
   and deterministic (high-count) subpopulations.
5. **Statistical model checking** — use Sequential Probability Ratio
   Test (SPRT) to estimate the probability that the switch stays in
   one state for a given duration.

**What this example does**
1. Builds a low-copy toggle switch (molecule counts, not concentrations).
2. Runs an SSA ensemble and computes statistics.
3. Runs moment closure approximation and compares to SSA.
4. Runs hybrid SSA/ODE simulation.
5. Uses SPRT-based statistical model checking to estimate the
   probability of remaining in the LacI-high state for 500 minutes.
"""

from __future__ import annotations

import sys
import numpy as np

# ── BioProver imports ──────────────────────────────────────────────────
from bioprover.models.bio_model import BioModel, Compartment
from bioprover.models.species import Species, SpeciesType
from bioprover.models.reactions import (
    HillRepression,
    LinearDegradation,
    Reaction,
    StoichiometryEntry,
)
from bioprover.models.parameters import Parameter, UncertaintyType

from bioprover.temporal.stl_ast import (
    Predicate,
    Expression,
    Always,
    Interval,
    ComparisonOp,
)

# Stochastic analysis modules
from bioprover.stochastic.ssa import (
    DirectMethod,
    TrajectoryRecorder,
    run_ensemble_ssa,
)
from bioprover.stochastic.ensemble import EnsembleSimulator, EnsembleStatistics
from bioprover.stochastic.moment_closure import (
    MomentEquations,
    NormalClosure,
    ClosureComparison,
)
from bioprover.stochastic.hybrid import (
    HaseltineRawlingsHybrid,
    SpeciesPartition,
)

# Statistical model checking
from bioprover.temporal.statistical_model_checking import (
    SPRTChecker,
    StatisticalModelChecker,
)


# ════════════════════════════════════════════════════════════════════════
#  Step 1 — Build a low-copy toggle switch
# ════════════════════════════════════════════════════════════════════════

def build_low_copy_toggle() -> BioModel:
    """Construct a toggle switch in the low-copy-number regime.

    Copy numbers are 10–50 molecules rather than nM concentrations.
    The Hill-function propensity is:
        a_prod(X_repressor) = α / (1 + (X_repressor / K)^n)
    where X_repressor is in molecule counts and K is scaled accordingly.

    Degradation is first-order with propensity δ·X.
    """
    model = BioModel(name="low_copy_toggle")
    model.add_compartment(Compartment("cell", size=1e-15))  # ~1 fL

    # Low copy numbers: LacI starts high, TetR starts low
    model.add_species(Species(
        name="LacI",
        species_type=SpeciesType.PROTEIN,
        initial_concentration=40.0,  # 40 molecules
        units="molecules",
        copy_number=40,
        compartment="cell",
    ))
    model.add_species(Species(
        name="TetR",
        species_type=SpeciesType.PROTEIN,
        initial_concentration=5.0,   # 5 molecules
        units="molecules",
        copy_number=5,
        compartment="cell",
    ))

    # Parameters scaled for molecule counts
    params = [
        Parameter("alpha1", value=50.0, units="molecules/min",
                  description="Max LacI production rate"),
        Parameter("alpha2", value=50.0, units="molecules/min",
                  description="Max TetR production rate"),
        Parameter("K1", value=20.0, units="molecules",
                  description="Half-repression for TetR → LacI"),
        Parameter("K2", value=20.0, units="molecules",
                  description="Half-repression for LacI → TetR"),
        Parameter("n", value=2.0, uncertainty_type=UncertaintyType.FIXED,
                  description="Hill coefficient"),
        Parameter("delta", value=0.07, units="1/min",
                  description="Degradation rate"),
    ]
    for p in params:
        model.add_parameter(p)

    # Reactions (same structure as deterministic model)
    model.add_reaction(Reaction(
        name="LacI_production",
        reactants=[],
        products=[StoichiometryEntry("LacI", 1)],
        kinetic_law=HillRepression(Vmax=50.0, K=20.0, n=2.0),
        modifiers=["TetR"],
    ))
    model.add_reaction(Reaction(
        name="TetR_production",
        reactants=[],
        products=[StoichiometryEntry("TetR", 1)],
        kinetic_law=HillRepression(Vmax=50.0, K=20.0, n=2.0),
        modifiers=["LacI"],
    ))
    model.add_reaction(Reaction(
        name="LacI_degradation",
        reactants=[StoichiometryEntry("LacI", 1)],
        products=[],
        kinetic_law=LinearDegradation(rate=0.07),
    ))
    model.add_reaction(Reaction(
        name="TetR_degradation",
        reactants=[StoichiometryEntry("TetR", 1)],
        products=[],
        kinetic_law=LinearDegradation(rate=0.07),
    ))

    return model


# ════════════════════════════════════════════════════════════════════════
#  Step 2 — SSA ensemble simulation
# ════════════════════════════════════════════════════════════════════════

def run_ssa_ensemble(model: BioModel, n_trajectories: int = 200,
                     t_end: float = 500.0):
    """Run Gillespie SSA ensemble and compute statistics.

    Each trajectory is an exact stochastic realisation of the chemical
    master equation.  From the ensemble we estimate means, variances,
    and the probability of switching states.
    """
    print("\n── SSA Ensemble Simulation ──")
    print(f"  Trajectories: {n_trajectories}")
    print(f"  Time horizon: {t_end} min")

    # Build SSA reaction list from the BioModel
    # run_ensemble_ssa accepts the model's reactions directly
    ssa_reactions = model.reactions
    initial_state = np.array([
        sp.initial_concentration for sp in model.species
    ], dtype=float)

    trajectories = run_ensemble_ssa(
        reactions=ssa_reactions,
        initial_state=initial_state,
        t_end=t_end,
        n_trajectories=n_trajectories,
        seed=42,
    )

    # Compute ensemble statistics via EnsembleSimulator
    ensemble_sim = EnsembleSimulator(
        reactions=ssa_reactions,
        num_species=model.num_species,
        method="direct",
    )

    # Summarise results
    print(f"  Completed {len(trajectories)} trajectories")

    # Count switching events: LacI drops below 15 (crossed to TetR-high)
    switch_threshold = 15.0
    n_switched = 0
    for traj in trajectories:
        # A trajectory is a list or array of states; check the final state
        if hasattr(traj, 'states') and len(traj.states) > 0:
            final_laci = traj.states[-1].get("LacI", traj.states[-1][0]
                                             if isinstance(traj.states[-1],
                                                           (list, np.ndarray))
                                             else 0)
            if final_laci < switch_threshold:
                n_switched += 1
        elif isinstance(traj, np.ndarray) and traj.ndim >= 1:
            if traj[-1] < switch_threshold:  # assume first column is LacI
                n_switched += 1

    switch_prob = n_switched / max(n_trajectories, 1)
    print(f"  Switching probability (t={t_end}): {switch_prob:.3f}")
    print(f"  Switched: {n_switched}/{n_trajectories}")

    return trajectories, switch_prob


# ════════════════════════════════════════════════════════════════════════
#  Step 3 — Moment closure approximation
# ════════════════════════════════════════════════════════════════════════

def run_moment_closure(model: BioModel, t_end: float = 500.0):
    """Approximate the stochastic dynamics via moment closure.

    Instead of simulating individual trajectories, moment closure
    derives ODEs for the *moments* (mean, variance, covariance) of
    the probability distribution.  A *closure* approximation (here:
    normal / Gaussian) truncates the infinite hierarchy of moment
    equations to make the system finite-dimensional.

    Advantages over SSA:
      - Much faster (single ODE solve vs thousands of trajectories).
      - Gives analytical expressions for mean and variance.
    Limitations:
      - Accuracy degrades for multi-modal distributions (e.g. the
        toggle switch in the bistable regime).
    """
    print("\n── Moment Closure Approximation ──")

    # Build moment equations from the model
    moment_eqs = MomentEquations.from_model(model)
    print(f"  Moment equations generated: {moment_eqs.num_equations} ODEs")

    # Apply normal (Gaussian) closure
    closure = NormalClosure()
    closed_system = closure.close(moment_eqs)
    print(f"  Closed system size: {closed_system.num_equations} ODEs")

    # Solve the closed moment ODEs
    t_span = (0.0, t_end)
    t, moments = closed_system.solve(t_span, num_points=200)

    # Extract mean and std for LacI (species index 0)
    mean_laci = moments[:, 0]   # E[LacI]
    var_laci  = moments[:, 2]   # Var[LacI] (index depends on ordering)
    std_laci  = np.sqrt(np.maximum(var_laci, 0.0))

    print(f"  t=0  : E[LacI]={mean_laci[0]:.1f}, "
          f"Std[LacI]={std_laci[0]:.1f}")
    print(f"  t={t_end:.0f}: E[LacI]={mean_laci[-1]:.1f}, "
          f"Std[LacI]={std_laci[-1]:.1f}")

    return t, mean_laci, std_laci


# ════════════════════════════════════════════════════════════════════════
#  Step 4 — Hybrid SSA/ODE simulation
# ════════════════════════════════════════════════════════════════════════

def run_hybrid(model: BioModel, t_end: float = 500.0):
    """Run hybrid stochastic-deterministic simulation.

    The Haseltine–Rawlings method partitions species into:
      - *Stochastic* species: low copy number, simulated by SSA.
      - *Deterministic* species: high copy number, simulated by ODE.

    For the toggle switch at low copy numbers, both species are
    stochastic, but this demonstrates the API.  In larger circuits
    with reporter proteins at high copy numbers, the hybrid approach
    gives significant speedups.
    """
    print("\n── Hybrid SSA/ODE Simulation ──")

    # Partition: both species are stochastic here (low count)
    partition = SpeciesPartition(
        stochastic_species=["LacI", "TetR"],
        deterministic_species=[],
    )

    hybrid = HaseltineRawlingsHybrid(
        model=model,
        partition=partition,
    )

    trajectory = hybrid.simulate(t_end=t_end, seed=123)

    # Report trajectory endpoints
    if hasattr(trajectory, 'times') and len(trajectory.times) > 0:
        print(f"  Hybrid trajectory: {len(trajectory.times)} time points")
        print(f"  Final state: t={trajectory.times[-1]:.1f}")
    else:
        print("  Hybrid trajectory completed")

    return trajectory


# ════════════════════════════════════════════════════════════════════════
#  Step 5 — Compare methods
# ════════════════════════════════════════════════════════════════════════

def compare_methods(ssa_switch_prob: float, moment_mean: np.ndarray,
                    moment_std: np.ndarray):
    """Compare SSA and moment closure results."""

    print("\n── Method Comparison ──")
    print(f"  SSA switching probability : {ssa_switch_prob:.3f}")
    print(f"  Moment closure final mean : {moment_mean[-1]:.1f}")
    print(f"  Moment closure final std  : {moment_std[-1]:.1f}")

    # The moment closure gives a unimodal approximation, so it cannot
    # capture the bimodal distribution of the toggle switch.  The SSA
    # ensemble reveals the true switching dynamics.
    if ssa_switch_prob > 0.01:
        print("  ⚠ SSA shows non-negligible switching — moment closure")
        print("    (unimodal) may underestimate this.")
    else:
        print("  ✓ Low switching probability — moment closure is a")
        print("    reasonable approximation for the dominant mode.")


# ════════════════════════════════════════════════════════════════════════
#  Step 6 — Statistical model checking (SPRT)
# ════════════════════════════════════════════════════════════════════════

def run_statistical_model_checking(model: BioModel):
    """Use SPRT to test: P(□[0,500] LacI > 15) ≥ 0.9.

    The Sequential Probability Ratio Test (SPRT) draws trajectories
    one at a time and decides between:
        H₀: P(φ) ≥ θ + ε       (property holds reliably)
        H₁: P(φ) ≤ θ − ε       (property fails too often)
    with configurable error bounds α (false positive) and β (false
    negative).  It is far more sample-efficient than naïve Monte Carlo
    when the true probability is far from the threshold.
    """
    print("\n── Statistical Model Checking (SPRT) ──")

    # Define the STL property: LacI stays above 15 for 500 min
    spec = Always(
        Predicate(Expression(variable="LacI"), ComparisonOp.GT, 15.0),
        Interval(0.0, 500.0),
    )
    print(f"  Property: □[0,500] (LacI > 15)")
    print(f"  Hypothesis threshold θ=0.9, indifference ε=0.05")
    print(f"  Error bounds: α=0.05, β=0.05")

    # Build a trajectory generator from the model's SSA reactions
    def trajectory_generator(seed=None):
        """Generate a single SSA trajectory."""
        initial = np.array([
            sp.initial_concentration for sp in model.species
        ], dtype=float)
        trajs = run_ensemble_ssa(
            reactions=model.reactions,
            initial_state=initial,
            t_end=500.0,
            n_trajectories=1,
            seed=seed,
        )
        return trajs[0] if trajs else None

    # Run SPRT
    sprt = SPRTChecker(
        formula=spec,
        generator=trajectory_generator,
        theta=0.9,          # null-hypothesis threshold
        indifference=0.05,  # half-width of indifference region
        alpha=0.05,         # type-I error bound
        beta=0.05,          # type-II error bound
        max_samples=500,
    )

    result = sprt.run()

    print(f"\n  Decision         : {result.decision.name}")
    print(f"  Samples used     : {result.num_samples}")
    print(f"  Satisfied        : {result.num_satisfied}")
    print(f"  Est. probability : {result.estimated_probability:.3f}")
    print(f"  95% CI           : [{result.lower_bound:.3f}, "
          f"{result.upper_bound:.3f}]")

    if result.decision.name == "ACCEPT":
        print("  ✓ High confidence: switch stays in LacI-high state")
        print("    with probability ≥ 0.85.")
    elif result.decision.name == "REJECT":
        print("  ✗ Switching probability is too high for reliable")
        print("    bistable memory at these copy numbers.")
    else:
        print("  ? Undecided — need more samples or narrower indifference.")

    return result


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  BioProver — Stochastic Toggle Switch Analysis")
    print("=" * 65)

    model = build_low_copy_toggle()
    print(f"\nModel : {model.name}")
    print(f"Species: {model.species_names}")
    print(f"Reactions: {model.num_reactions}")
    print(f"Initial copies: LacI=40, TetR=5")

    # 1. SSA ensemble
    trajectories, switch_prob = run_ssa_ensemble(
        model, n_trajectories=200, t_end=500.0
    )

    # 2. Moment closure
    t_mc, mean_laci, std_laci = run_moment_closure(model, t_end=500.0)

    # 3. Hybrid simulation
    hybrid_traj = run_hybrid(model, t_end=500.0)

    # 4. Compare
    compare_methods(switch_prob, mean_laci, std_laci)

    # 5. Statistical model checking
    sprt_result = run_statistical_model_checking(model)

    print(f"\n{'═' * 65}")
    print("  Analysis complete.")
    print(f"{'═' * 65}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
