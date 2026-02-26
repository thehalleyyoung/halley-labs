"""Benchmark suites for evaluating the BioProver CEGAR verification engine.

Provides a library of synthetic biology circuit benchmarks with known
verification outcomes, spanning toggle switches, repressilators,
feed-forward loops, cascades and logic gates.  Each benchmark pairs a
:class:`BioModel` with an STL specification and an expected
:class:`VerificationStatus`, enabling automated regression and
performance testing of the CEGAR pipeline.
"""

from __future__ import annotations

import logging
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bioprover.cegar import (
    CEGARConfig,
    CEGAREngine,
    CEGARStatistics,
    VerificationResult,
    VerificationStatus,
)
from bioprover.models import (
    BioModel,
    Compartment,
    HillActivation,
    HillRepression,
    LinearDegradation,
    ConstitutiveProduction,
    MassAction,
    Parameter,
    ParameterSet,
    Reaction,
    Species,
    SpeciesType,
    UncertaintyType,
)
from bioprover.models.reactions import StoichiometryEntry
from bioprover.spec import SpecificationTemplate, TemplateLibrary
from bioprover.temporal import (
    Always,
    Eventually,
    Interval,
    Predicate,
    STLAnd,
    STLFormula,
    STLNot,
    STLOr,
    Until,
)
from bioprover.temporal.stl_ast import ComparisonOp, make_var_expr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & data-classes
# ---------------------------------------------------------------------------


class BenchmarkDifficulty(Enum):
    """Coarse difficulty classification for benchmark circuits."""

    EASY = auto()
    MEDIUM = auto()
    HARD = auto()
    FRONTIER = auto()


@dataclass
class BenchmarkCircuit:
    """A single benchmark instance pairing a model with its specification."""

    model: BioModel
    specification: STLFormula
    expected_result: VerificationStatus
    difficulty: BenchmarkDifficulty
    name: str
    description: str
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.metadata:
            self.metadata = {
                "species_count": len(self.model.species)
                if hasattr(self.model, "species")
                else 0,
                "parameter_count": len(self.model.parameters)
                if hasattr(self.model, "parameters")
                else 0,
                "time_horizon": _extract_time_horizon(self.specification),
            }


@dataclass
class BenchmarkResult:
    """Outcome of running a single :class:`BenchmarkCircuit`."""

    benchmark: BenchmarkCircuit
    actual_result: Optional[VerificationResult]
    wall_time: float
    peak_memory_mb: float
    correct: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _extract_time_horizon(formula: STLFormula) -> float:
    """Recursively find the largest time-bound in an STL formula tree."""
    horizon = 0.0
    if isinstance(formula, (Always, Eventually)):
        horizon = max(horizon, formula.interval.hi)
        horizon = max(horizon, _extract_time_horizon(formula.child))
    elif isinstance(formula, Until):
        horizon = max(horizon, formula.interval.hi)
        horizon = max(horizon, _extract_time_horizon(formula.left))
        horizon = max(horizon, _extract_time_horizon(formula.right))
    elif isinstance(formula, (STLAnd, STLOr)):
        horizon = max(horizon, _extract_time_horizon(formula.left))
        horizon = max(horizon, _extract_time_horizon(formula.right))
    elif isinstance(formula, STLNot):
        horizon = max(horizon, _extract_time_horizon(formula.child))
    return horizon


def _hill_param(name: str, value: float, low: float, high: float,
                units: str = "nM") -> Parameter:
    """Create a Hill-function parameter with uniform uncertainty."""
    return Parameter(
        name=name, value=value, units=units,
        lower_bound=low, upper_bound=high,
        uncertainty_type=UncertaintyType.UNIFORM,
    )


def _rate_param(name: str, value: float, low: float, high: float,
                units: str = "1/min") -> Parameter:
    """Create a rate parameter with uniform uncertainty."""
    return Parameter(
        name=name, value=value, units=units,
        lower_bound=low, upper_bound=high,
        uncertainty_type=UncertaintyType.UNIFORM,
    )


def _pred_ge(species_name: str, threshold: float) -> Predicate:
    """Shorthand: species concentration >= threshold."""
    return Predicate(
        expr=make_var_expr(species_name),
        op=ComparisonOp.GE,
        threshold=threshold,
    )


def _pred_le(species_name: str, threshold: float) -> Predicate:
    """Shorthand: species concentration <= threshold."""
    return Predicate(
        expr=make_var_expr(species_name),
        op=ComparisonOp.LE,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Built-in benchmark generators
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Collection of built-in and parameterised benchmark circuits.

    Use the class methods to obtain individual benchmarks or call
    :meth:`all_benchmarks` for the full suite.
    """

    # -- Toggle switch --------------------------------------------------

    @staticmethod
    def toggle_switch() -> BenchmarkCircuit:
        """Gardner-Collins bistable toggle switch (LacI / TetR / IPTG).

        Three species, eight parameters.  The specification asserts
        bistability: starting from a high-LacI state the system must
        remain high-LacI, and likewise for TetR.
        """
        model = BioModel(name="toggle_switch")
        model.add_species(Species("LacI", initial_concentration=100.0,
                                  species_type=SpeciesType.PROTEIN))
        model.add_species(Species("TetR", initial_concentration=10.0,
                                  species_type=SpeciesType.PROTEIN))
        model.add_species(Species("IPTG", initial_concentration=0.0,
                                  species_type=SpeciesType.SMALL_MOLECULE))

        params = [
            _hill_param("alpha_LacI", 5.0, 1.0, 10.0, "nM/min"),
            _hill_param("alpha_TetR", 5.0, 1.0, 10.0, "nM/min"),
            _hill_param("K_LacI", 200.0, 100.0, 400.0),
            _hill_param("K_TetR", 200.0, 100.0, 400.0),
            _hill_param("n_LacI", 2.0, 1.5, 4.0, ""),
            _hill_param("n_TetR", 2.0, 1.5, 4.0, ""),
            _rate_param("deg_LacI", 0.05, 0.01, 0.1),
            _rate_param("deg_TetR", 0.05, 0.01, 0.1),
        ]
        for p in params:
            model.parameters.add(p)

        model.add_reaction(Reaction(
            "prod_LacI", [], [StoichiometryEntry("LacI")],
            HillRepression(Vmax=5.0, K=200.0, n=2.0),
        ))
        model.add_reaction(Reaction(
            "prod_TetR", [], [StoichiometryEntry("TetR")],
            HillRepression(Vmax=5.0, K=200.0, n=2.0),
        ))
        model.add_reaction(Reaction(
            "deg_LacI", [StoichiometryEntry("LacI")], [],
            LinearDegradation(rate=0.05),
        ))
        model.add_reaction(Reaction(
            "deg_TetR", [StoichiometryEntry("TetR")], [],
            LinearDegradation(rate=0.05),
        ))

        # Bistability: LacI stays high (>50 nM) for all t in [10, 200]
        spec = Always(
            STLAnd(_pred_ge("LacI", 50.0), _pred_le("TetR", 30.0)),
            Interval(10.0, 200.0),
        )

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.VERIFIED,
            difficulty=BenchmarkDifficulty.EASY,
            name="toggle_switch",
            description="Gardner-Collins toggle switch bistability",
            category="bistable",
            tags=["classic", "bistable", "toggle"],
        )

    # -- Repressilator --------------------------------------------------

    @staticmethod
    def repressilator() -> BenchmarkCircuit:
        """Elowitz-Leibler three-gene repressilator.

        Five species (3 proteins, 2 additional mRNAs acting as
        intermediates in the ring).  Twelve parameters.  The
        specification asserts sustained oscillation via alternating
        Eventually operators.
        """
        model = BioModel(name="repressilator")
        proteins = ["LacI", "TetR", "CI"]
        mrnas = ["mRNA_LacI", "mRNA_TetR"]
        for prot in proteins:
            model.add_species(Species(prot, initial_concentration=20.0,
                                      species_type=SpeciesType.PROTEIN))
        for m in mrnas:
            model.add_species(Species(m, initial_concentration=5.0,
                                      species_type=SpeciesType.MRNA))

        repression_pairs = [
            ("LacI", "TetR"), ("TetR", "CI"), ("CI", "LacI"),
        ]
        idx = 0
        for repressor, target in repression_pairs:
            idx += 1
            vmax = 3.0 + idx * 0.5
            K = 150.0 + idx * 50.0
            n = 2.5
            model.parameters.add(
                _hill_param(f"Vmax_{target}", vmax, 1.0, 8.0, "nM/min"))
            model.parameters.add(_hill_param(f"K_{target}", K, 50.0, 500.0))
            model.parameters.add(_hill_param(f"n_{target}", n, 1.5, 4.0, ""))
            model.parameters.add(
                _rate_param(f"deg_{target}", 0.04, 0.01, 0.08))

            model.add_reaction(Reaction(
                f"prod_{target}", [], [StoichiometryEntry(target)],
                HillRepression(Vmax=vmax, K=K, n=n),
            ))
            model.add_reaction(Reaction(
                f"deg_{target}", [StoichiometryEntry(target)], [],
                LinearDegradation(rate=0.04),
            ))

        # Oscillation: LacI eventually rises above 80, then eventually
        # drops below 20, repeatedly over [0, 500].
        rise = Eventually(_pred_ge("LacI", 80.0), Interval(0.0, 250.0))
        fall = Eventually(_pred_le("LacI", 20.0), Interval(0.0, 250.0))
        spec = Always(STLAnd(rise, fall), Interval(0.0, 500.0))

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.VERIFIED,
            difficulty=BenchmarkDifficulty.MEDIUM,
            name="repressilator",
            description="Three-gene repressilator oscillation",
            category="oscillator",
            tags=["classic", "oscillator", "ring"],
        )

    # -- Feed-forward loop ----------------------------------------------

    @staticmethod
    def feed_forward_loop(loop_type: str = "C1-I1") -> BenchmarkCircuit:
        """Coherent / incoherent feed-forward loop motif.

        Parameters
        ----------
        loop_type:
            One of ``C1-I1`` through ``C1-I4`` (coherent) or ``C2-I1``
            through ``C2-I4`` (incoherent).
        """
        coherent = loop_type.startswith("C1")
        variant = int(loop_type.split("-I")[1]) if "-I" in loop_type else 1

        model = BioModel(name=f"ffl_{loop_type}")
        for sp in ["X", "Y", "Z"]:
            model.add_species(Species(sp, initial_concentration=0.0,
                                      species_type=SpeciesType.PROTEIN))
        model.add_species(Species("Signal", initial_concentration=0.0,
                                  species_type=SpeciesType.SMALL_MOLECULE))

        K_base = 100.0 + 50.0 * variant
        n_hill = 2.0 + 0.5 * (variant - 1)
        vmax_xy = 4.0
        vmax_xz = 3.0 if coherent else 5.0

        for pname, val, lo, hi, u in [
            ("K_XY", K_base, 50.0, 400.0, "nM"),
            ("K_XZ", K_base + 50.0, 50.0, 500.0, "nM"),
            ("n_XY", n_hill, 1.5, 4.0, ""),
            ("n_XZ", n_hill, 1.5, 4.0, ""),
            ("Vmax_XY", vmax_xy, 1.0, 8.0, "nM/min"),
            ("Vmax_XZ", vmax_xz, 1.0, 8.0, "nM/min"),
            ("deg_Y", 0.06, 0.02, 0.1, "1/min"),
            ("deg_Z", 0.06, 0.02, 0.1, "1/min"),
        ]:
            model.parameters.add(_hill_param(pname, val, lo, hi, u))

        # X -> Y (always activation)
        model.add_reaction(Reaction(
            "prod_Y", [], [StoichiometryEntry("Y")],
            HillActivation(Vmax=vmax_xy, K=K_base, n=n_hill),
        ))
        # X -> Z (activation or repression)
        if coherent:
            model.add_reaction(Reaction(
                "prod_Z_direct", [], [StoichiometryEntry("Z")],
                HillActivation(Vmax=vmax_xz, K=K_base + 50.0, n=n_hill),
            ))
        else:
            model.add_reaction(Reaction(
                "prod_Z_direct", [], [StoichiometryEntry("Z")],
                HillRepression(Vmax=vmax_xz, K=K_base + 50.0, n=n_hill),
            ))

        # Y -> Z (activation in all variants)
        model.add_reaction(Reaction(
            "prod_Z_indirect", [], [StoichiometryEntry("Z")],
            HillActivation(Vmax=3.0, K=200.0, n=2.0),
        ))
        for sp in ["Y", "Z"]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=0.06),
            ))

        if coherent:
            # Pulse filtering: Z should eventually rise and stay above
            # a threshold
            spec = Eventually(
                Always(_pred_ge("Z", 30.0), Interval(0.0, 100.0)),
                Interval(0.0, 300.0),
            )
            expected = VerificationStatus.VERIFIED
        else:
            # Pulse generation: Z rises then falls
            rise = Eventually(_pred_ge("Z", 40.0), Interval(0.0, 150.0))
            fall = Eventually(_pred_le("Z", 10.0), Interval(100.0, 300.0))
            spec = STLAnd(rise, fall)
            expected = VerificationStatus.BOUNDED_GUARANTEE

        diff = (BenchmarkDifficulty.EASY if variant <= 2
                else BenchmarkDifficulty.MEDIUM)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=expected,
            difficulty=diff,
            name=f"ffl_{loop_type}",
            description=f"Feed-forward loop type {loop_type}",
            category="feed_forward",
            tags=["motif", "ffl", loop_type.lower()],
        )

    # -- Cascade --------------------------------------------------------

    @staticmethod
    def cascade(depth: int = 3) -> BenchmarkCircuit:
        """Linear signalling cascade with *depth* stages (2–5)."""
        depth = max(2, min(depth, 5))
        return BenchmarkSuite.n_stage_cascade(depth)

    # -- Logic gates ----------------------------------------------------

    @staticmethod
    def nand_gate() -> BenchmarkCircuit:
        """Biological NAND gate (two-input, one-output)."""
        model = BioModel(name="nand_gate")
        for sp in ["A", "B", "Out"]:
            model.add_species(Species(sp, initial_concentration=0.0,
                                      species_type=SpeciesType.PROTEIN))
        model.parameters.add(_hill_param("K_A", 150.0, 50.0, 300.0))
        model.parameters.add(_hill_param("K_B", 150.0, 50.0, 300.0))
        model.parameters.add(_hill_param("n", 2.0, 1.5, 3.0, ""))
        model.parameters.add(
            _hill_param("Vmax_Out", 6.0, 2.0, 10.0, "nM/min"))
        model.parameters.add(_rate_param("deg_Out", 0.05, 0.01, 0.1))

        # Out is repressed by both A and B (NAND logic)
        model.add_reaction(Reaction(
            "prod_Out", [], [StoichiometryEntry("Out")],
            HillRepression(Vmax=6.0, K=150.0, n=2.0),
        ))
        model.add_reaction(Reaction(
            "deg_Out", [StoichiometryEntry("Out")], [],
            LinearDegradation(rate=0.05),
        ))

        # NAND truth: when both inputs high, output low
        both_high = STLAnd(_pred_ge("A", 80.0), _pred_ge("B", 80.0))
        out_low = _pred_le("Out", 20.0)
        spec = Always(
            STLOr(STLNot(both_high), out_low),
            Interval(20.0, 300.0),
        )

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.VERIFIED,
            difficulty=BenchmarkDifficulty.EASY,
            name="nand_gate",
            description="Two-input biological NAND gate correctness",
            category="logic_gate",
            tags=["logic", "nand", "digital"],
        )

    @staticmethod
    def nor_gate() -> BenchmarkCircuit:
        """Biological NOR gate (two-input, one-output)."""
        model = BioModel(name="nor_gate")
        for sp in ["A", "B", "Out"]:
            model.add_species(Species(sp, initial_concentration=0.0,
                                      species_type=SpeciesType.PROTEIN))
        model.parameters.add(_hill_param("K_A", 120.0, 50.0, 250.0))
        model.parameters.add(_hill_param("K_B", 120.0, 50.0, 250.0))
        model.parameters.add(_hill_param("n", 3.0, 2.0, 4.0, ""))
        model.parameters.add(
            _hill_param("Vmax_Out", 7.0, 3.0, 12.0, "nM/min"))
        model.parameters.add(_rate_param("deg_Out", 0.04, 0.01, 0.08))

        model.add_reaction(Reaction(
            "prod_Out", [], [StoichiometryEntry("Out")],
            HillRepression(Vmax=7.0, K=120.0, n=3.0),
        ))
        model.add_reaction(Reaction(
            "deg_Out", [StoichiometryEntry("Out")], [],
            LinearDegradation(rate=0.04),
        ))

        # NOR truth: when either input high, output low
        any_high = STLOr(_pred_ge("A", 80.0), _pred_ge("B", 80.0))
        out_low = _pred_le("Out", 15.0)
        spec = Always(
            STLOr(STLNot(any_high), out_low),
            Interval(20.0, 300.0),
        )

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.VERIFIED,
            difficulty=BenchmarkDifficulty.EASY,
            name="nor_gate",
            description="Two-input biological NOR gate correctness",
            category="logic_gate",
            tags=["logic", "nor", "digital"],
        )

    # ----------------------------------------------------------------
    # Scalable circuit generators
    # ----------------------------------------------------------------

    @staticmethod
    def n_stage_cascade(n: int) -> BenchmarkCircuit:
        """Parameterised *n*-stage signalling cascade.

        Each stage is a Hill-activated production fed by the previous
        stage, with first-order degradation.  The specification checks
        that the signal propagates to the final stage within a
        time-horizon proportional to *n*.
        """
        if n < 2:
            raise ValueError("Cascade must have at least 2 stages")

        model = BioModel(name=f"cascade_{n}")
        stage_names = [f"S{i}" for i in range(n)]

        # First stage gets constitutive input
        model.add_species(Species(
            stage_names[0], initial_concentration=80.0,
            species_type=SpeciesType.PROTEIN,
        ))
        for i in range(1, n):
            model.add_species(Species(
                stage_names[i], initial_concentration=0.0,
                species_type=SpeciesType.PROTEIN,
            ))

        for i in range(1, n):
            vmax = 5.0 + 0.5 * i
            K = 100.0 + 30.0 * i
            n_hill = 2.0
            deg = 0.03 + 0.005 * i

            model.parameters.add(
                _hill_param(f"Vmax_{stage_names[i]}", vmax, 2.0, 10.0,
                            "nM/min"))
            model.parameters.add(
                _hill_param(f"K_{stage_names[i]}", K, 50.0, 400.0))
            model.parameters.add(
                _hill_param(f"n_{stage_names[i]}", n_hill, 1.5, 3.0, ""))
            model.parameters.add(
                _rate_param(f"deg_{stage_names[i]}", deg, 0.01, 0.08))

            model.add_reaction(Reaction(
                f"prod_{stage_names[i]}", [],
                [StoichiometryEntry(stage_names[i])],
                HillActivation(Vmax=vmax, K=K, n=n_hill),
            ))
            model.add_reaction(Reaction(
                f"deg_{stage_names[i]}",
                [StoichiometryEntry(stage_names[i])], [],
                LinearDegradation(rate=deg),
            ))

        horizon = 100.0 * n
        spec = Eventually(
            _pred_ge(stage_names[-1], 40.0),
            Interval(0.0, horizon),
        )

        diff = {
            2: BenchmarkDifficulty.EASY,
            3: BenchmarkDifficulty.MEDIUM,
            4: BenchmarkDifficulty.HARD,
        }.get(n, BenchmarkDifficulty.FRONTIER)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.VERIFIED,
            difficulty=diff,
            name=f"cascade_{n}",
            description=f"{n}-stage signalling cascade propagation",
            category="cascade",
            tags=["cascade", "scalable", f"depth_{n}"],
        )

    @staticmethod
    def n_node_repressilator(n: int) -> BenchmarkCircuit:
        """Ring oscillator with *n* repressors (n must be odd, ≥ 3)."""
        if n < 3 or n % 2 == 0:
            raise ValueError("Ring oscillator requires odd n >= 3")

        model = BioModel(name=f"repressilator_{n}")
        node_names = [f"G{i}" for i in range(n)]

        for name in node_names:
            model.add_species(Species(
                name, initial_concentration=10.0 + 5.0 * random.Random(42).random(),
                species_type=SpeciesType.PROTEIN,
            ))

        for i in range(n):
            target = node_names[i]
            vmax = 4.0
            K = 200.0
            n_hill = 2.5
            deg = 0.04

            model.parameters.add(
                _hill_param(f"Vmax_{target}", vmax, 1.5, 8.0, "nM/min"))
            model.parameters.add(
                _hill_param(f"K_{target}", K, 80.0, 400.0))
            model.parameters.add(
                _hill_param(f"n_{target}", n_hill, 1.5, 4.0, ""))
            model.parameters.add(
                _rate_param(f"deg_{target}", deg, 0.01, 0.08))

            model.add_reaction(Reaction(
                f"prod_{target}", [],
                [StoichiometryEntry(target)],
                HillRepression(Vmax=vmax, K=K, n=n_hill),
            ))
            model.add_reaction(Reaction(
                f"deg_{target}", [StoichiometryEntry(target)], [],
                LinearDegradation(rate=deg),
            ))

        horizon = 150.0 * n
        rise = Eventually(_pred_ge(node_names[0], 70.0),
                          Interval(0.0, horizon / 2))
        fall = Eventually(_pred_le(node_names[0], 15.0),
                          Interval(0.0, horizon / 2))
        spec = Always(STLAnd(rise, fall), Interval(0.0, horizon))

        diff = {3: BenchmarkDifficulty.MEDIUM, 5: BenchmarkDifficulty.HARD}
        difficulty = diff.get(n, BenchmarkDifficulty.FRONTIER)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.BOUNDED_GUARANTEE,
            difficulty=difficulty,
            name=f"repressilator_{n}",
            description=f"{n}-node ring oscillator",
            category="oscillator",
            tags=["oscillator", "ring", "scalable", f"nodes_{n}"],
        )

    @staticmethod
    def random_grn(
        n_species: int = 6,
        n_reactions: int = 10,
        connectivity: float = 0.3,
        seed: int = 42,
    ) -> BenchmarkCircuit:
        """Generate a random gene regulatory network.

        Parameters
        ----------
        n_species:
            Number of protein species.
        n_reactions:
            Number of regulatory interactions (activation / repression).
        connectivity:
            Edge probability used when *n_reactions* exceeds the
            connectivity-implied count.
        seed:
            RNG seed for reproducibility.
        """
        rng = random.Random(seed)
        model = BioModel(name=f"random_grn_s{n_species}_r{n_reactions}")

        species_names = [f"P{i}" for i in range(n_species)]
        for sp in species_names:
            model.add_species(Species(
                sp, initial_concentration=rng.uniform(0.0, 50.0),
                species_type=SpeciesType.PROTEIN,
            ))

        edges_added = 0
        attempted: set[Tuple[int, int]] = set()
        while edges_added < n_reactions:
            src = rng.randint(0, n_species - 1)
            tgt = rng.randint(0, n_species - 1)
            if src == tgt or (src, tgt) in attempted:
                if len(attempted) >= n_species * (n_species - 1):
                    break
                continue
            attempted.add((src, tgt))

            vmax = rng.uniform(2.0, 8.0)
            K = rng.uniform(80.0, 400.0)
            n_hill = rng.uniform(1.5, 4.0)
            is_activation = rng.random() > 0.5

            tag = "act" if is_activation else "rep"
            rxn_name = f"{tag}_{species_names[src]}_{species_names[tgt]}"
            model.parameters.add(_hill_param(
                f"Vmax_{rxn_name}", vmax, 1.0, 10.0, "nM/min"))
            model.parameters.add(
                _hill_param(f"K_{rxn_name}", K, 50.0, 500.0))
            model.parameters.add(
                _hill_param(f"n_{rxn_name}", n_hill, 1.0, 5.0, ""))

            law: Any = (HillActivation if is_activation
                        else HillRepression)(Vmax=vmax, K=K, n=n_hill)
            model.add_reaction(Reaction(
                rxn_name, [], [StoichiometryEntry(species_names[tgt])], law,
            ))
            edges_added += 1

        for sp in species_names:
            deg = rng.uniform(0.02, 0.08)
            model.parameters.add(_rate_param(f"deg_{sp}", deg, 0.01, 0.1))
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=deg),
            ))

        # Reachability spec on first species
        spec = Eventually(
            _pred_ge(species_names[0], 30.0),
            Interval(0.0, 500.0),
        )

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.UNKNOWN,
            difficulty=BenchmarkDifficulty.HARD,
            name=f"random_grn_{n_species}_{n_reactions}",
            description=(f"Random GRN with {n_species} species and "
                         f"{n_reactions} interactions"),
            category="random_grn",
            tags=["random", "grn", "scalable"],
        )

    # ----------------------------------------------------------------
    # Large-scale benchmark circuits (10+ species)
    # ----------------------------------------------------------------

    @staticmethod
    def mapk_cascade() -> BenchmarkCircuit:
        """MAPK-like phosphorylation cascade with 8 species.

        Linear cascade: Input activates MKKK* via Hill function, each
        kinase activates the next, with phosphatases providing negative
        regulation.  Models ultrasensitive signal amplification.
        """
        model = BioModel(name="mapk_cascade")
        species_list = [
            ("Input", 100.0, SpeciesType.SMALL_MOLECULE),
            ("MKKKs", 0.0, SpeciesType.PROTEIN),   # MKKK*
            ("MKK", 80.0, SpeciesType.PROTEIN),
            ("MKKs", 0.0, SpeciesType.PROTEIN),     # MKK*
            ("MK", 80.0, SpeciesType.PROTEIN),
            ("MKs", 0.0, SpeciesType.PROTEIN),       # MK*
            ("Phosphatase1", 20.0, SpeciesType.PROTEIN),
            ("Phosphatase2", 20.0, SpeciesType.PROTEIN),
        ]
        for name, ic, stype in species_list:
            model.add_species(Species(name, initial_concentration=ic,
                                      species_type=stype))

        # Parameters for the cascade
        cascade_params = [
            _hill_param("Vmax_MKKKs", 6.0, 2.0, 12.0, "nM/min"),
            _hill_param("K_MKKKs", 100.0, 50.0, 300.0),
            _hill_param("n_MKKKs", 3.0, 2.0, 5.0, ""),
            _hill_param("Vmax_MKKs", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_MKKs", 120.0, 50.0, 350.0),
            _hill_param("n_MKKs", 2.5, 1.5, 4.0, ""),
            _hill_param("Vmax_MKs", 4.5, 1.5, 9.0, "nM/min"),
            _hill_param("K_MKs", 150.0, 60.0, 400.0),
            _hill_param("n_MKs", 2.0, 1.5, 3.5, ""),
            _rate_param("deg_MKKKs", 0.03, 0.01, 0.06),
            _rate_param("deg_MKKs", 0.04, 0.01, 0.08),
            _rate_param("deg_MKs", 0.05, 0.02, 0.1),
            _rate_param("dephos_MKKKs", 0.02, 0.005, 0.05),
            _rate_param("dephos_MKKs", 0.025, 0.005, 0.06),
            _rate_param("dephos_MKs", 0.03, 0.01, 0.07),
        ]
        for p in cascade_params:
            model.parameters.add(p)

        # Input -> MKKK* (Hill activation)
        model.add_reaction(Reaction(
            "prod_MKKKs", [], [StoichiometryEntry("MKKKs")],
            HillActivation(Vmax=6.0, K=100.0, n=3.0),
        ))
        # MKKK* -> MKK* (Hill activation)
        model.add_reaction(Reaction(
            "prod_MKKs", [], [StoichiometryEntry("MKKs")],
            HillActivation(Vmax=5.0, K=120.0, n=2.5),
        ))
        # MKK* -> MK* (Hill activation)
        model.add_reaction(Reaction(
            "prod_MKs", [], [StoichiometryEntry("MKs")],
            HillActivation(Vmax=4.5, K=150.0, n=2.0),
        ))
        # Degradation of activated forms
        for sp, rate in [("MKKKs", 0.03), ("MKKs", 0.04), ("MKs", 0.05)]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=rate),
            ))
        # Phosphatase-mediated dephosphorylation (modeled as degradation)
        for sp, rate in [("MKKKs", 0.02), ("MKKs", 0.025), ("MKs", 0.03)]:
            model.add_reaction(Reaction(
                f"dephos_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=rate),
            ))

        # Safety: MK* stays bounded; Liveness: MK* eventually rises
        safety = Always(_pred_le("MKs", 200.0), Interval(0.0, 500.0))
        liveness = Eventually(_pred_ge("MKs", 30.0), Interval(0.0, 300.0))
        spec = STLAnd(safety, liveness)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.VERIFIED,
            difficulty=BenchmarkDifficulty.HARD,
            name="mapk_cascade",
            description="MAPK-like 8-species phosphorylation cascade",
            category="cascade",
            tags=["cascade", "mapk", "phosphorylation", "large"],
            metadata={
                "species_count": 8, "parameter_count": 15,
                "expected_time_s": 120,
            },
        )

    @staticmethod
    def quorum_sensing() -> BenchmarkCircuit:
        """Multicellular quorum sensing circuit with 12 species.

        Two cells each contain [LuxI, AHL_int, LuxR, LuxR_AHL, GFP]
        and share an external AHL pool.  The spec checks eventual
        synchronization of GFP levels between cells.
        """
        model = BioModel(name="quorum_sensing")

        cell_species = ["LuxI", "AHL_int", "LuxR", "LuxR_AHL", "GFP"]
        # Two cells plus shared AHL
        for cell_id in [1, 2]:
            for sp in cell_species:
                ic = 10.0 if sp == "LuxR" else 0.0
                stype = (SpeciesType.PROTEIN if sp != "AHL_int"
                         else SpeciesType.SMALL_MOLECULE)
                model.add_species(Species(
                    f"{sp}_c{cell_id}", initial_concentration=ic,
                    species_type=stype,
                ))
        # Shared external AHL (cell 1 starts with slight asymmetry)
        model.add_species(Species(
            "AHL_ext", initial_concentration=0.0,
            species_type=SpeciesType.SMALL_MOLECULE,
        ))
        # Additional shared species for total AHL tracking
        model.add_species(Species(
            "AHL_ext_total", initial_concentration=0.0,
            species_type=SpeciesType.SMALL_MOLECULE,
        ))

        # Parameters
        qs_params = [
            _hill_param("Vmax_LuxI", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_LuxI", 100.0, 40.0, 250.0),
            _hill_param("n_LuxI", 2.0, 1.5, 3.0, ""),
            _rate_param("k_AHL_prod", 0.1, 0.02, 0.3),
            _rate_param("k_AHL_diff", 0.05, 0.01, 0.15),
            _rate_param("k_bind_LuxR", 0.01, 0.002, 0.05),
            _hill_param("Vmax_GFP", 8.0, 3.0, 15.0, "nM/min"),
            _hill_param("K_GFP", 80.0, 30.0, 200.0),
            _hill_param("n_GFP", 2.0, 1.5, 3.5, ""),
            _rate_param("deg_LuxI", 0.03, 0.01, 0.06),
            _rate_param("deg_AHL", 0.02, 0.005, 0.05),
            _rate_param("deg_LuxR", 0.01, 0.005, 0.03),
            _rate_param("deg_GFP", 0.02, 0.01, 0.05),
            _rate_param("deg_LuxR_AHL", 0.015, 0.005, 0.04),
        ]
        for p in qs_params:
            model.parameters.add(p)

        for cell_id in [1, 2]:
            c = f"_c{cell_id}"
            # LuxR_AHL activates LuxI production (positive feedback)
            model.add_reaction(Reaction(
                f"prod_LuxI{c}", [], [StoichiometryEntry(f"LuxI{c}")],
                HillActivation(Vmax=5.0, K=100.0, n=2.0),
            ))
            # LuxI produces internal AHL
            model.add_reaction(Reaction(
                f"prod_AHL_int{c}", [], [StoichiometryEntry(f"AHL_int{c}")],
                ConstitutiveProduction(rate=0.1),
            ))
            # AHL diffuses out (modeled as degradation of internal + production of external)
            model.add_reaction(Reaction(
                f"AHL_diff_out{c}",
                [StoichiometryEntry(f"AHL_int{c}")], [],
                LinearDegradation(rate=0.05),
            ))
            model.add_reaction(Reaction(
                f"AHL_diff_in{c}", [], [StoichiometryEntry(f"AHL_int{c}")],
                ConstitutiveProduction(rate=0.05),
            ))
            # LuxR + AHL_int -> LuxR_AHL (modeled as production)
            model.add_reaction(Reaction(
                f"bind_LuxR{c}", [], [StoichiometryEntry(f"LuxR_AHL{c}")],
                MassAction(k_forward=0.01),
            ))
            # LuxR_AHL activates GFP
            model.add_reaction(Reaction(
                f"prod_GFP{c}", [], [StoichiometryEntry(f"GFP{c}")],
                HillActivation(Vmax=8.0, K=80.0, n=2.0),
            ))
            # Degradation reactions
            for sp, rate in [("LuxI", 0.03), ("AHL_int", 0.02),
                             ("LuxR", 0.01), ("GFP", 0.02),
                             ("LuxR_AHL", 0.015)]:
                model.add_reaction(Reaction(
                    f"deg_{sp}{c}", [StoichiometryEntry(f"{sp}{c}")], [],
                    LinearDegradation(rate=rate),
                ))
            # Constitutive LuxR production
            model.add_reaction(Reaction(
                f"const_LuxR{c}", [], [StoichiometryEntry(f"LuxR{c}")],
                ConstitutiveProduction(rate=0.5),
            ))

        # External AHL dynamics
        model.add_reaction(Reaction(
            "deg_AHL_ext", [StoichiometryEntry("AHL_ext")], [],
            LinearDegradation(rate=0.01),
        ))

        # Synchronization spec: eventually GFP levels in both cells
        # are close (both above 20)
        sync = Eventually(
            STLAnd(_pred_ge("GFP_c1", 20.0), _pred_ge("GFP_c2", 20.0)),
            Interval(0.0, 500.0),
        )
        # Both GFPs bounded
        bounded = Always(
            STLAnd(_pred_le("GFP_c1", 300.0), _pred_le("GFP_c2", 300.0)),
            Interval(0.0, 500.0),
        )
        spec = STLAnd(sync, bounded)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.BOUNDED_GUARANTEE,
            difficulty=BenchmarkDifficulty.FRONTIER,
            name="quorum_sensing",
            description="12-species multicellular quorum sensing synchronization",
            category="multicellular",
            tags=["quorum", "multicellular", "synchronization", "large"],
            metadata={
                "species_count": 12, "parameter_count": 14,
                "expected_time_s": 300,
            },
        )

    @staticmethod
    def genetic_clock_reporter() -> BenchmarkCircuit:
        """Repressilator coupled to a downstream reporter cascade (10 species).

        Core repressilator (TetR, LacI, CI) with mRNA intermediates
        (mRNA_TetR, mRNA_LacI, mRNA_CI) driving a reporter cascade
        (Reporter1 -> Reporter2 -> Reporter3 -> GFP).
        """
        model = BioModel(name="genetic_clock_reporter")

        species_data = [
            ("TetR", 30.0, SpeciesType.PROTEIN),
            ("LacI", 10.0, SpeciesType.PROTEIN),
            ("CI", 20.0, SpeciesType.PROTEIN),
            ("mRNA_TetR", 5.0, SpeciesType.MRNA),
            ("mRNA_LacI", 5.0, SpeciesType.MRNA),
            ("mRNA_CI", 5.0, SpeciesType.MRNA),
            ("Reporter1", 0.0, SpeciesType.PROTEIN),
            ("Reporter2", 0.0, SpeciesType.PROTEIN),
            ("Reporter3", 0.0, SpeciesType.PROTEIN),
            ("GFP", 0.0, SpeciesType.PROTEIN),
        ]
        for name, ic, stype in species_data:
            model.add_species(Species(name, initial_concentration=ic,
                                      species_type=stype))

        clock_params = [
            # Repressilator core
            _hill_param("Vmax_mRNA_TetR", 4.0, 1.5, 8.0, "nM/min"),
            _hill_param("K_mRNA_TetR", 180.0, 80.0, 400.0),
            _hill_param("n_mRNA_TetR", 2.5, 1.5, 4.0, ""),
            _hill_param("Vmax_mRNA_LacI", 4.0, 1.5, 8.0, "nM/min"),
            _hill_param("K_mRNA_LacI", 180.0, 80.0, 400.0),
            _hill_param("n_mRNA_LacI", 2.5, 1.5, 4.0, ""),
            _hill_param("Vmax_mRNA_CI", 4.0, 1.5, 8.0, "nM/min"),
            _hill_param("K_mRNA_CI", 180.0, 80.0, 400.0),
            _hill_param("n_mRNA_CI", 2.5, 1.5, 4.0, ""),
            # Translation rates
            _rate_param("k_transl_TetR", 0.5, 0.2, 1.0),
            _rate_param("k_transl_LacI", 0.5, 0.2, 1.0),
            _rate_param("k_transl_CI", 0.5, 0.2, 1.0),
            # Reporter cascade
            _hill_param("Vmax_Rep1", 3.0, 1.0, 6.0, "nM/min"),
            _hill_param("K_Rep1", 100.0, 40.0, 250.0),
            _hill_param("Vmax_Rep2", 3.0, 1.0, 6.0, "nM/min"),
            _hill_param("K_Rep2", 100.0, 40.0, 250.0),
            _hill_param("Vmax_Rep3", 3.0, 1.0, 6.0, "nM/min"),
            _hill_param("K_Rep3", 100.0, 40.0, 250.0),
            _hill_param("Vmax_GFP", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_GFP", 80.0, 30.0, 200.0),
            # Degradation
            _rate_param("deg_mRNA", 0.1, 0.04, 0.2),
            _rate_param("deg_prot", 0.04, 0.01, 0.08),
            _rate_param("deg_reporter", 0.03, 0.01, 0.06),
            _rate_param("deg_GFP", 0.02, 0.01, 0.05),
        ]
        for p in clock_params:
            model.parameters.add(p)

        # Repressilator: CI --| TetR, TetR --| LacI, LacI --| CI (via mRNA)
        repression_map = [
            ("CI", "mRNA_TetR", 4.0, 180.0, 2.5),
            ("TetR", "mRNA_LacI", 4.0, 180.0, 2.5),
            ("LacI", "mRNA_CI", 4.0, 180.0, 2.5),
        ]
        for repressor, target_mrna, vmax, K, n in repression_map:
            model.add_reaction(Reaction(
                f"prod_{target_mrna}", [], [StoichiometryEntry(target_mrna)],
                HillRepression(Vmax=vmax, K=K, n=n),
            ))

        # Translation: mRNA -> protein
        for mrna, prot in [("mRNA_TetR", "TetR"), ("mRNA_LacI", "LacI"),
                           ("mRNA_CI", "CI")]:
            model.add_reaction(Reaction(
                f"transl_{prot}", [], [StoichiometryEntry(prot)],
                ConstitutiveProduction(rate=0.5),
            ))

        # Reporter cascade: CI activates Reporter1, each reporter
        # activates the next
        reporter_chain = [
            ("Reporter1", 3.0, 100.0),
            ("Reporter2", 3.0, 100.0),
            ("Reporter3", 3.0, 100.0),
        ]
        for rep, vmax, K in reporter_chain:
            model.add_reaction(Reaction(
                f"prod_{rep}", [], [StoichiometryEntry(rep)],
                HillActivation(Vmax=vmax, K=K, n=2.0),
            ))
        # Reporter3 activates GFP
        model.add_reaction(Reaction(
            "prod_GFP", [], [StoichiometryEntry("GFP")],
            HillActivation(Vmax=5.0, K=80.0, n=2.0),
        ))

        # Degradation for all species
        for sp in ["mRNA_TetR", "mRNA_LacI", "mRNA_CI"]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=0.1),
            ))
        for sp in ["TetR", "LacI", "CI"]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=0.04),
            ))
        for sp in ["Reporter1", "Reporter2", "Reporter3"]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=0.03),
            ))
        model.add_reaction(Reaction(
            "deg_GFP", [StoichiometryEntry("GFP")], [],
            LinearDegradation(rate=0.02),
        ))

        # Oscillation in GFP: rises and falls
        rise = Eventually(_pred_ge("GFP", 30.0), Interval(0.0, 400.0))
        fall = Eventually(_pred_le("GFP", 5.0), Interval(0.0, 400.0))
        spec = Always(STLAnd(rise, fall), Interval(0.0, 800.0))

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.BOUNDED_GUARANTEE,
            difficulty=BenchmarkDifficulty.HARD,
            name="genetic_clock_reporter",
            description="10-species repressilator with downstream reporter cascade",
            category="oscillator",
            tags=["oscillator", "repressilator", "reporter", "large"],
            metadata={
                "species_count": 10, "parameter_count": 24,
                "expected_time_s": 180,
            },
        )

    @staticmethod
    def metabolic_toggle_feedback() -> BenchmarkCircuit:
        """Extended toggle switch with metabolic feedback (15 species).

        Two competing pathways, each with 3 enzymes and 2 metabolites,
        plus a shared Precursor.  Crosstalk and resource competition
        introduce complex dynamics.
        """
        model = BioModel(name="metabolic_toggle_feedback")

        species_data = [
            # Pathway A: enzymes and metabolites
            ("EnzA1", 10.0, SpeciesType.PROTEIN),
            ("EnzA2", 5.0, SpeciesType.PROTEIN),
            ("EnzA3", 2.0, SpeciesType.PROTEIN),
            ("MetA1", 0.0, SpeciesType.SMALL_MOLECULE),
            ("MetA2", 0.0, SpeciesType.SMALL_MOLECULE),
            # Pathway B: enzymes and metabolites
            ("EnzB1", 5.0, SpeciesType.PROTEIN),
            ("EnzB2", 10.0, SpeciesType.PROTEIN),
            ("EnzB3", 2.0, SpeciesType.PROTEIN),
            ("MetB1", 0.0, SpeciesType.SMALL_MOLECULE),
            ("MetB2", 0.0, SpeciesType.SMALL_MOLECULE),
            # Shared resources
            ("Precursor", 50.0, SpeciesType.SMALL_MOLECULE),
            ("Resource", 100.0, SpeciesType.SMALL_MOLECULE),
            # Toggle regulators
            ("RegA", 20.0, SpeciesType.PROTEIN),
            ("RegB", 10.0, SpeciesType.PROTEIN),
            # Output reporter
            ("Output", 0.0, SpeciesType.PROTEIN),
        ]
        for name, ic, stype in species_data:
            model.add_species(Species(name, initial_concentration=ic,
                                      species_type=stype))

        toggle_params = [
            # Toggle core
            _hill_param("Vmax_RegA", 6.0, 2.0, 12.0, "nM/min"),
            _hill_param("K_RegA", 150.0, 60.0, 350.0),
            _hill_param("n_RegA", 2.5, 1.5, 4.0, ""),
            _hill_param("Vmax_RegB", 6.0, 2.0, 12.0, "nM/min"),
            _hill_param("K_RegB", 150.0, 60.0, 350.0),
            _hill_param("n_RegB", 2.5, 1.5, 4.0, ""),
            # Enzyme expression
            _hill_param("Vmax_EnzA", 4.0, 1.5, 8.0, "nM/min"),
            _hill_param("K_EnzA", 100.0, 40.0, 250.0),
            _hill_param("Vmax_EnzB", 4.0, 1.5, 8.0, "nM/min"),
            _hill_param("K_EnzB", 100.0, 40.0, 250.0),
            # Metabolic rates
            _rate_param("k_metA1", 0.08, 0.02, 0.2),
            _rate_param("k_metA2", 0.06, 0.02, 0.15),
            _rate_param("k_metB1", 0.08, 0.02, 0.2),
            _rate_param("k_metB2", 0.06, 0.02, 0.15),
            # Resource competition
            _rate_param("k_resource_A", 0.01, 0.002, 0.03),
            _rate_param("k_resource_B", 0.01, 0.002, 0.03),
            # Precursor supply and degradation
            _rate_param("k_precursor_prod", 0.5, 0.2, 1.0),
            _rate_param("deg_precursor", 0.01, 0.005, 0.03),
            # General degradation
            _rate_param("deg_enz", 0.03, 0.01, 0.06),
            _rate_param("deg_met", 0.05, 0.02, 0.1),
            _rate_param("deg_reg", 0.04, 0.01, 0.08),
            _rate_param("deg_output", 0.02, 0.01, 0.05),
            # Feedback strength
            _hill_param("Vmax_feedback_A", 2.0, 0.5, 5.0, "nM/min"),
            _hill_param("K_feedback_A", 50.0, 20.0, 150.0),
            _hill_param("Vmax_feedback_B", 2.0, 0.5, 5.0, "nM/min"),
            _hill_param("K_feedback_B", 50.0, 20.0, 150.0),
            _hill_param("Vmax_Output", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_Output", 80.0, 30.0, 200.0),
        ]
        for p in toggle_params:
            model.parameters.add(p)

        # Toggle core: RegB --| RegA, RegA --| RegB
        model.add_reaction(Reaction(
            "prod_RegA", [], [StoichiometryEntry("RegA")],
            HillRepression(Vmax=6.0, K=150.0, n=2.5),
        ))
        model.add_reaction(Reaction(
            "prod_RegB", [], [StoichiometryEntry("RegB")],
            HillRepression(Vmax=6.0, K=150.0, n=2.5),
        ))

        # RegA activates pathway A enzymes
        for enz in ["EnzA1", "EnzA2", "EnzA3"]:
            model.add_reaction(Reaction(
                f"prod_{enz}", [], [StoichiometryEntry(enz)],
                HillActivation(Vmax=4.0, K=100.0, n=2.0),
            ))
        # RegB activates pathway B enzymes
        for enz in ["EnzB1", "EnzB2", "EnzB3"]:
            model.add_reaction(Reaction(
                f"prod_{enz}", [], [StoichiometryEntry(enz)],
                HillActivation(Vmax=4.0, K=100.0, n=2.0),
            ))

        # Metabolite production (simplified: enzyme converts precursor)
        for met, rate in [("MetA1", 0.08), ("MetA2", 0.06),
                          ("MetB1", 0.08), ("MetB2", 0.06)]:
            model.add_reaction(Reaction(
                f"prod_{met}", [], [StoichiometryEntry(met)],
                ConstitutiveProduction(rate=rate),
            ))

        # Metabolic feedback: MetA2 enhances RegA, MetB2 enhances RegB
        model.add_reaction(Reaction(
            "feedback_A", [], [StoichiometryEntry("RegA")],
            HillActivation(Vmax=2.0, K=50.0, n=2.0),
        ))
        model.add_reaction(Reaction(
            "feedback_B", [], [StoichiometryEntry("RegB")],
            HillActivation(Vmax=2.0, K=50.0, n=2.0),
        ))

        # Resource and precursor dynamics
        model.add_reaction(Reaction(
            "prod_Precursor", [], [StoichiometryEntry("Precursor")],
            ConstitutiveProduction(rate=0.5),
        ))
        model.add_reaction(Reaction(
            "deg_Precursor", [StoichiometryEntry("Precursor")], [],
            LinearDegradation(rate=0.01),
        ))
        model.add_reaction(Reaction(
            "consume_Resource_A", [StoichiometryEntry("Resource")], [],
            LinearDegradation(rate=0.01),
        ))
        model.add_reaction(Reaction(
            "consume_Resource_B", [StoichiometryEntry("Resource")], [],
            LinearDegradation(rate=0.01),
        ))

        # Output driven by MetA2 (pathway A output)
        model.add_reaction(Reaction(
            "prod_Output", [], [StoichiometryEntry("Output")],
            HillActivation(Vmax=5.0, K=80.0, n=2.0),
        ))

        # Degradation
        for sp in ["EnzA1", "EnzA2", "EnzA3", "EnzB1", "EnzB2", "EnzB3"]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=0.03),
            ))
        for sp in ["MetA1", "MetA2", "MetB1", "MetB2"]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=0.05),
            ))
        for sp in ["RegA", "RegB"]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=0.04),
            ))
        model.add_reaction(Reaction(
            "deg_Output", [StoichiometryEntry("Output")], [],
            LinearDegradation(rate=0.02),
        ))
        model.add_reaction(Reaction(
            "deg_Resource", [StoichiometryEntry("Resource")], [],
            LinearDegradation(rate=0.005),
        ))

        # Bistability: one pathway dominates; check RegA stays high OR RegB stays high
        state_a = Always(_pred_ge("RegA", 40.0), Interval(50.0, 400.0))
        state_b = Always(_pred_ge("RegB", 40.0), Interval(50.0, 400.0))
        bounded_out = Always(_pred_le("Output", 300.0), Interval(0.0, 400.0))
        spec = STLAnd(STLOr(state_a, state_b), bounded_out)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.BOUNDED_GUARANTEE,
            difficulty=BenchmarkDifficulty.FRONTIER,
            name="metabolic_toggle_feedback",
            description="15-species metabolic toggle with feedback and resource competition",
            category="bistable",
            tags=["toggle", "metabolic", "feedback", "resource_competition",
                  "large"],
            metadata={
                "species_count": 15, "parameter_count": 28,
                "expected_time_s": 600,
            },
        )

    @staticmethod
    def biosensor_amplification() -> BenchmarkCircuit:
        """Biosensor with amplification cascade (10 species).

        Input analyte → Sensor → three-stage amplification cascade →
        Reporter → GFP.  Includes noise species for signal-to-noise
        modeling.
        """
        model = BioModel(name="biosensor_amplification")

        species_data = [
            ("Analyte", 50.0, SpeciesType.SMALL_MOLECULE),
            ("Sensor", 0.0, SpeciesType.PROTEIN),
            ("Amp1", 0.0, SpeciesType.PROTEIN),
            ("Amp2", 0.0, SpeciesType.PROTEIN),
            ("Amp3", 0.0, SpeciesType.PROTEIN),
            ("Reporter", 0.0, SpeciesType.PROTEIN),
            ("GFP", 0.0, SpeciesType.PROTEIN),
            ("Noise1", 5.0, SpeciesType.SMALL_MOLECULE),
            ("Noise2", 3.0, SpeciesType.SMALL_MOLECULE),
            ("Background", 2.0, SpeciesType.PROTEIN),
        ]
        for name, ic, stype in species_data:
            model.add_species(Species(name, initial_concentration=ic,
                                      species_type=stype))

        bio_params = [
            # Sensor binding
            _hill_param("Vmax_Sensor", 6.0, 2.0, 12.0, "nM/min"),
            _hill_param("K_Sensor", 30.0, 10.0, 80.0),
            _hill_param("n_Sensor", 1.5, 1.0, 3.0, ""),
            # Amplification cascade
            _hill_param("Vmax_Amp1", 8.0, 3.0, 15.0, "nM/min"),
            _hill_param("K_Amp1", 50.0, 20.0, 120.0),
            _hill_param("n_Amp1", 2.0, 1.5, 3.0, ""),
            _hill_param("Vmax_Amp2", 10.0, 4.0, 18.0, "nM/min"),
            _hill_param("K_Amp2", 40.0, 15.0, 100.0),
            _hill_param("n_Amp2", 2.0, 1.5, 3.0, ""),
            _hill_param("Vmax_Amp3", 12.0, 5.0, 20.0, "nM/min"),
            _hill_param("K_Amp3", 35.0, 10.0, 90.0),
            _hill_param("n_Amp3", 2.5, 1.5, 4.0, ""),
            # Reporter and GFP
            _hill_param("Vmax_Reporter", 7.0, 3.0, 14.0, "nM/min"),
            _hill_param("K_Reporter", 60.0, 25.0, 150.0),
            _hill_param("Vmax_GFP", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_GFP", 50.0, 20.0, 120.0),
            # Noise and background
            _rate_param("k_noise1", 0.1, 0.02, 0.3),
            _rate_param("k_noise2", 0.08, 0.02, 0.2),
            _rate_param("k_background", 0.05, 0.01, 0.15),
            # Degradation rates
            _rate_param("deg_Sensor", 0.05, 0.02, 0.1),
            _rate_param("deg_Amp", 0.04, 0.01, 0.08),
            _rate_param("deg_Reporter", 0.03, 0.01, 0.06),
            _rate_param("deg_GFP", 0.02, 0.01, 0.05),
            _rate_param("deg_Noise", 0.06, 0.02, 0.12),
        ]
        for p in bio_params:
            model.parameters.add(p)

        # Analyte activates Sensor
        model.add_reaction(Reaction(
            "prod_Sensor", [], [StoichiometryEntry("Sensor")],
            HillActivation(Vmax=6.0, K=30.0, n=1.5),
        ))
        # Amplification cascade: each stage activates the next with
        # increasing gain
        amp_stages = [
            ("Amp1", 8.0, 50.0, 2.0),
            ("Amp2", 10.0, 40.0, 2.0),
            ("Amp3", 12.0, 35.0, 2.5),
        ]
        for name, vmax, K, n in amp_stages:
            model.add_reaction(Reaction(
                f"prod_{name}", [], [StoichiometryEntry(name)],
                HillActivation(Vmax=vmax, K=K, n=n),
            ))
        # Amp3 activates Reporter
        model.add_reaction(Reaction(
            "prod_Reporter", [], [StoichiometryEntry("Reporter")],
            HillActivation(Vmax=7.0, K=60.0, n=2.0),
        ))
        # Reporter activates GFP
        model.add_reaction(Reaction(
            "prod_GFP", [], [StoichiometryEntry("GFP")],
            HillActivation(Vmax=5.0, K=50.0, n=2.0),
        ))
        # Noise sources (constitutive, low-level)
        model.add_reaction(Reaction(
            "prod_Noise1", [], [StoichiometryEntry("Noise1")],
            ConstitutiveProduction(rate=0.1),
        ))
        model.add_reaction(Reaction(
            "prod_Noise2", [], [StoichiometryEntry("Noise2")],
            ConstitutiveProduction(rate=0.08),
        ))
        model.add_reaction(Reaction(
            "prod_Background", [], [StoichiometryEntry("Background")],
            ConstitutiveProduction(rate=0.05),
        ))

        # Degradation
        model.add_reaction(Reaction(
            "deg_Sensor", [StoichiometryEntry("Sensor")], [],
            LinearDegradation(rate=0.05),
        ))
        for sp in ["Amp1", "Amp2", "Amp3"]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=0.04),
            ))
        model.add_reaction(Reaction(
            "deg_Reporter", [StoichiometryEntry("Reporter")], [],
            LinearDegradation(rate=0.03),
        ))
        model.add_reaction(Reaction(
            "deg_GFP", [StoichiometryEntry("GFP")], [],
            LinearDegradation(rate=0.02),
        ))
        for sp in ["Noise1", "Noise2"]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=0.06),
            ))
        model.add_reaction(Reaction(
            "deg_Background", [StoichiometryEntry("Background")], [],
            LinearDegradation(rate=0.04),
        ))
        # Analyte degrades slowly
        model.add_reaction(Reaction(
            "deg_Analyte", [StoichiometryEntry("Analyte")], [],
            LinearDegradation(rate=0.005),
        ))

        # Signal detection: GFP eventually exceeds noise background
        signal_detected = Eventually(
            _pred_ge("GFP", 50.0), Interval(0.0, 300.0),
        )
        # Signal-to-noise: GFP >> Background
        snr = Always(
            STLOr(_pred_le("Background", 20.0), _pred_ge("GFP", 30.0)),
            Interval(50.0, 300.0),
        )
        spec = STLAnd(signal_detected, snr)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.VERIFIED,
            difficulty=BenchmarkDifficulty.HARD,
            name="biosensor_amplification",
            description="10-species biosensor with signal amplification cascade",
            category="biosensor",
            tags=["biosensor", "amplification", "cascade", "snr", "large"],
            metadata={
                "species_count": 10, "parameter_count": 24,
                "expected_time_s": 150,
            },
        )

    # ----------------------------------------------------------------
    # Additional large-scale benchmark circuits (10-20 species)
    # ----------------------------------------------------------------

    @staticmethod
    def mapk_3tier_cascade() -> BenchmarkCircuit:
        """MAPK 3-tier cascade (11 species, 22 parameters).

        Classic signaling cascade with three tiers of kinase/phosphatase
        pairs: MAPKKK, MAPKK, MAPK. Each tier has inactive/active forms
        and a dedicated phosphatase.  Input signal activates MAPKKK.
        """
        model = BioModel(name="mapk_3tier_cascade")

        species_data = [
            ("Signal", 100.0, SpeciesType.SMALL_MOLECULE),
            ("MAPKKK", 80.0, SpeciesType.PROTEIN),
            ("MAPKKK_act", 0.0, SpeciesType.PROTEIN),
            ("MAPKK", 80.0, SpeciesType.PROTEIN),
            ("MAPKK_act", 0.0, SpeciesType.PROTEIN),
            ("MAPK", 80.0, SpeciesType.PROTEIN),
            ("MAPK_act", 0.0, SpeciesType.PROTEIN),
            ("Phos1", 15.0, SpeciesType.PROTEIN),
            ("Phos2", 15.0, SpeciesType.PROTEIN),
            ("Phos3", 15.0, SpeciesType.PROTEIN),
            ("Output", 0.0, SpeciesType.PROTEIN),
        ]
        for name, ic, stype in species_data:
            model.add_species(Species(name, initial_concentration=ic,
                                       species_type=stype))

        params = [
            _hill_param("Vmax_KKK_act", 8.0, 3.0, 15.0, "nM/min"),
            _hill_param("K_KKK_act", 80.0, 30.0, 200.0),
            _hill_param("n_KKK_act", 3.0, 2.0, 5.0, ""),
            _hill_param("Vmax_KK_act", 6.0, 2.0, 12.0, "nM/min"),
            _hill_param("K_KK_act", 100.0, 40.0, 250.0),
            _hill_param("n_KK_act", 2.5, 1.5, 4.0, ""),
            _hill_param("Vmax_K_act", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_K_act", 120.0, 50.0, 300.0),
            _hill_param("n_K_act", 2.0, 1.5, 3.5, ""),
            _hill_param("Vmax_Output", 4.0, 1.5, 8.0, "nM/min"),
            _hill_param("K_Output", 60.0, 25.0, 150.0),
            _hill_param("n_Output", 2.0, 1.5, 3.0, ""),
            _rate_param("deg_KKK_act", 0.03, 0.01, 0.06),
            _rate_param("deg_KK_act", 0.04, 0.01, 0.08),
            _rate_param("deg_K_act", 0.05, 0.02, 0.1),
            _rate_param("dephos_KKK", 0.025, 0.005, 0.06),
            _rate_param("dephos_KK", 0.03, 0.01, 0.07),
            _rate_param("dephos_K", 0.035, 0.01, 0.08),
            _rate_param("deg_Output", 0.02, 0.01, 0.05),
            _rate_param("deg_Signal", 0.001, 0.0005, 0.005),
            _rate_param("deg_Phos", 0.01, 0.005, 0.02),
            _rate_param("k_basal", 0.01, 0.002, 0.03),
        ]
        for p in params:
            model.parameters.add(p)

        # Signal -> MAPKKK_act
        model.add_reaction(Reaction(
            "act_MAPKKK", [], [StoichiometryEntry("MAPKKK_act")],
            HillActivation(Vmax=8.0, K=80.0, n=3.0),
        ))
        # MAPKKK_act -> MAPKK_act
        model.add_reaction(Reaction(
            "act_MAPKK", [], [StoichiometryEntry("MAPKK_act")],
            HillActivation(Vmax=6.0, K=100.0, n=2.5),
        ))
        # MAPKK_act -> MAPK_act
        model.add_reaction(Reaction(
            "act_MAPK", [], [StoichiometryEntry("MAPK_act")],
            HillActivation(Vmax=5.0, K=120.0, n=2.0),
        ))
        # MAPK_act -> Output
        model.add_reaction(Reaction(
            "prod_Output", [], [StoichiometryEntry("Output")],
            HillActivation(Vmax=4.0, K=60.0, n=2.0),
        ))
        # Degradation and dephosphorylation
        for sp, rate in [("MAPKKK_act", 0.03), ("MAPKK_act", 0.04),
                          ("MAPK_act", 0.05)]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=rate),
            ))
            model.add_reaction(Reaction(
                f"dephos_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=rate * 0.8),
            ))
        model.add_reaction(Reaction(
            "deg_Output", [StoichiometryEntry("Output")], [],
            LinearDegradation(rate=0.02),
        ))

        # Ultrasensitive response: Output eventually rises; bounded
        safety = Always(_pred_le("MAPK_act", 200.0), Interval(0.0, 600.0))
        liveness = Eventually(_pred_ge("Output", 25.0), Interval(0.0, 400.0))
        spec = STLAnd(safety, liveness)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.VERIFIED,
            difficulty=BenchmarkDifficulty.HARD,
            name="mapk_3tier_cascade",
            description="11-species MAPK 3-tier signaling cascade",
            category="cascade",
            tags=["cascade", "mapk", "ultrasensitive", "signaling", "large"],
            metadata={
                "species_count": 11, "parameter_count": 22,
                "expected_time_s": 200,
            },
        )

    @staticmethod
    def genetic_oscillator_reporter() -> BenchmarkCircuit:
        """Genetic oscillator with reporter (10 species, 24 parameters).

        Activator-repressor oscillator with mRNA intermediates and a
        downstream GFP reporter cascade.  Based on Atkinson et al. (2003).
        """
        model = BioModel(name="genetic_oscillator_reporter")

        species_data = [
            ("ActProt", 20.0, SpeciesType.PROTEIN),
            ("RepProt", 10.0, SpeciesType.PROTEIN),
            ("mRNA_Act", 5.0, SpeciesType.MRNA),
            ("mRNA_Rep", 5.0, SpeciesType.MRNA),
            ("ActProt_dim", 0.0, SpeciesType.PROTEIN),
            ("RepProt_dim", 0.0, SpeciesType.PROTEIN),
            ("Reporter1", 0.0, SpeciesType.PROTEIN),
            ("Reporter2", 0.0, SpeciesType.PROTEIN),
            ("GFP", 0.0, SpeciesType.PROTEIN),
            ("Inducer", 50.0, SpeciesType.SMALL_MOLECULE),
        ]
        for name, ic, stype in species_data:
            model.add_species(Species(name, initial_concentration=ic,
                                       species_type=stype))

        params = [
            _hill_param("Vmax_mRNA_Act", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_mRNA_Act", 150.0, 60.0, 350.0),
            _hill_param("n_mRNA_Act", 2.0, 1.5, 3.5, ""),
            _hill_param("Vmax_mRNA_Rep", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_mRNA_Rep", 150.0, 60.0, 350.0),
            _hill_param("n_mRNA_Rep", 2.0, 1.5, 3.5, ""),
            _rate_param("k_transl_Act", 0.5, 0.2, 1.0),
            _rate_param("k_transl_Rep", 0.5, 0.2, 1.0),
            _rate_param("k_dim_Act", 0.01, 0.003, 0.03),
            _rate_param("k_dim_Rep", 0.01, 0.003, 0.03),
            _hill_param("Vmax_Rep1", 3.0, 1.0, 6.0, "nM/min"),
            _hill_param("K_Rep1", 80.0, 30.0, 200.0),
            _hill_param("Vmax_Rep2", 3.0, 1.0, 6.0, "nM/min"),
            _hill_param("K_Rep2", 80.0, 30.0, 200.0),
            _hill_param("Vmax_GFP", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_GFP", 60.0, 25.0, 150.0),
            _rate_param("deg_mRNA_Act", 0.1, 0.04, 0.2),
            _rate_param("deg_mRNA_Rep", 0.1, 0.04, 0.2),
            _rate_param("deg_ActProt", 0.04, 0.01, 0.08),
            _rate_param("deg_RepProt", 0.04, 0.01, 0.08),
            _rate_param("deg_dim", 0.02, 0.005, 0.05),
            _rate_param("deg_reporter", 0.03, 0.01, 0.06),
            _rate_param("deg_GFP", 0.02, 0.01, 0.05),
            _rate_param("deg_Inducer", 0.005, 0.001, 0.01),
        ]
        for p in params:
            model.parameters.add(p)

        # Activator activates its own mRNA (positive feedback)
        model.add_reaction(Reaction(
            "prod_mRNA_Act", [], [StoichiometryEntry("mRNA_Act")],
            HillActivation(Vmax=5.0, K=150.0, n=2.0),
        ))
        # Repressor represses activator mRNA
        model.add_reaction(Reaction(
            "repress_mRNA_Act", [StoichiometryEntry("mRNA_Act")], [],
            LinearDegradation(rate=0.05),
        ))
        # Activator activates repressor mRNA
        model.add_reaction(Reaction(
            "prod_mRNA_Rep", [], [StoichiometryEntry("mRNA_Rep")],
            HillActivation(Vmax=5.0, K=150.0, n=2.0),
        ))
        # Translation
        for mrna, prot in [("mRNA_Act", "ActProt"), ("mRNA_Rep", "RepProt")]:
            model.add_reaction(Reaction(
                f"transl_{prot}", [], [StoichiometryEntry(prot)],
                ConstitutiveProduction(rate=0.5),
            ))
        # Dimerization
        for prot, dimer in [("ActProt", "ActProt_dim"),
                            ("RepProt", "RepProt_dim")]:
            model.add_reaction(Reaction(
                f"dim_{prot}", [], [StoichiometryEntry(dimer)],
                MassAction(k_forward=0.01),
            ))
        # Reporter cascade: ActProt -> Reporter1 -> Reporter2 -> GFP
        for rep, vmax, K in [("Reporter1", 3.0, 80.0),
                             ("Reporter2", 3.0, 80.0)]:
            model.add_reaction(Reaction(
                f"prod_{rep}", [], [StoichiometryEntry(rep)],
                HillActivation(Vmax=vmax, K=K, n=2.0),
            ))
        model.add_reaction(Reaction(
            "prod_GFP", [], [StoichiometryEntry("GFP")],
            HillActivation(Vmax=5.0, K=60.0, n=2.0),
        ))
        # Degradation
        for sp, rate in [("mRNA_Act", 0.1), ("mRNA_Rep", 0.1),
                          ("ActProt", 0.04), ("RepProt", 0.04),
                          ("ActProt_dim", 0.02), ("RepProt_dim", 0.02),
                          ("Reporter1", 0.03), ("Reporter2", 0.03),
                          ("GFP", 0.02), ("Inducer", 0.005)]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=rate),
            ))

        rise = Eventually(_pred_ge("GFP", 25.0), Interval(0.0, 400.0))
        fall = Eventually(_pred_le("GFP", 5.0), Interval(0.0, 400.0))
        spec = Always(STLAnd(rise, fall), Interval(0.0, 800.0))

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.BOUNDED_GUARANTEE,
            difficulty=BenchmarkDifficulty.HARD,
            name="genetic_oscillator_reporter",
            description="10-species genetic oscillator with GFP reporter",
            category="oscillator",
            tags=["oscillator", "activator_repressor", "reporter", "large"],
            metadata={
                "species_count": 10, "parameter_count": 24,
                "expected_time_s": 200,
            },
        )

    @staticmethod
    def quorum_sensing_extended() -> BenchmarkCircuit:
        """Quorum sensing circuit (12 species, 14 parameters).

        Single-cell quorum sensing with LuxI/LuxR/AHL signaling
        pathway, including intracellular and extracellular AHL pools,
        LuxR-AHL complex formation, and dual reporter outputs.
        """
        model = BioModel(name="quorum_sensing_extended")

        species_data = [
            ("LuxI", 0.0, SpeciesType.PROTEIN),
            ("LuxR", 10.0, SpeciesType.PROTEIN),
            ("AHL_int", 0.0, SpeciesType.SMALL_MOLECULE),
            ("AHL_ext", 0.0, SpeciesType.SMALL_MOLECULE),
            ("LuxR_AHL", 0.0, SpeciesType.PROTEIN),
            ("GFP", 0.0, SpeciesType.PROTEIN),
            ("RFP", 0.0, SpeciesType.PROTEIN),
            ("mRNA_LuxI", 2.0, SpeciesType.MRNA),
            ("mRNA_LuxR", 5.0, SpeciesType.MRNA),
            ("mRNA_GFP", 0.0, SpeciesType.MRNA),
            ("LasR", 5.0, SpeciesType.PROTEIN),
            ("LasI", 0.0, SpeciesType.PROTEIN),
        ]
        for name, ic, stype in species_data:
            model.add_species(Species(name, initial_concentration=ic,
                                       species_type=stype))

        params = [
            _hill_param("Vmax_LuxI", 5.0, 2.0, 10.0, "nM/min"),
            _hill_param("K_LuxI", 100.0, 40.0, 250.0),
            _hill_param("n_LuxI", 2.0, 1.5, 3.0, ""),
            _rate_param("k_AHL_prod", 0.1, 0.02, 0.3),
            _rate_param("k_AHL_diff", 0.05, 0.01, 0.15),
            _rate_param("k_bind", 0.01, 0.002, 0.05),
            _hill_param("Vmax_GFP", 8.0, 3.0, 15.0, "nM/min"),
            _hill_param("K_GFP", 80.0, 30.0, 200.0),
            _rate_param("deg_LuxI", 0.03, 0.01, 0.06),
            _rate_param("deg_AHL", 0.02, 0.005, 0.05),
            _rate_param("deg_LuxR", 0.01, 0.005, 0.03),
            _rate_param("deg_GFP", 0.02, 0.01, 0.05),
            _rate_param("deg_RFP", 0.025, 0.01, 0.05),
            _rate_param("deg_complex", 0.015, 0.005, 0.04),
        ]
        for p in params:
            model.parameters.add(p)

        # LuxR_AHL activates LuxI production
        model.add_reaction(Reaction(
            "prod_LuxI", [], [StoichiometryEntry("LuxI")],
            HillActivation(Vmax=5.0, K=100.0, n=2.0),
        ))
        # LuxI produces AHL
        model.add_reaction(Reaction(
            "prod_AHL_int", [], [StoichiometryEntry("AHL_int")],
            ConstitutiveProduction(rate=0.1),
        ))
        # AHL diffusion
        model.add_reaction(Reaction(
            "AHL_diff_out", [StoichiometryEntry("AHL_int")], [],
            LinearDegradation(rate=0.05),
        ))
        model.add_reaction(Reaction(
            "AHL_diff_in", [], [StoichiometryEntry("AHL_int")],
            ConstitutiveProduction(rate=0.05),
        ))
        # LuxR binding
        model.add_reaction(Reaction(
            "bind_LuxR", [], [StoichiometryEntry("LuxR_AHL")],
            MassAction(k_forward=0.01),
        ))
        # Reporters
        model.add_reaction(Reaction(
            "prod_GFP", [], [StoichiometryEntry("GFP")],
            HillActivation(Vmax=8.0, K=80.0, n=2.0),
        ))
        model.add_reaction(Reaction(
            "prod_RFP", [], [StoichiometryEntry("RFP")],
            HillActivation(Vmax=6.0, K=100.0, n=2.0),
        ))
        # Constitutive LuxR and LasR
        model.add_reaction(Reaction(
            "const_LuxR", [], [StoichiometryEntry("LuxR")],
            ConstitutiveProduction(rate=0.5),
        ))
        model.add_reaction(Reaction(
            "const_LasR", [], [StoichiometryEntry("LasR")],
            ConstitutiveProduction(rate=0.3),
        ))
        model.add_reaction(Reaction(
            "prod_LasI", [], [StoichiometryEntry("LasI")],
            ConstitutiveProduction(rate=0.05),
        ))
        # mRNA production
        for mrna in ["mRNA_LuxI", "mRNA_LuxR", "mRNA_GFP"]:
            model.add_reaction(Reaction(
                f"prod_{mrna}", [], [StoichiometryEntry(mrna)],
                ConstitutiveProduction(rate=0.3),
            ))
        # Degradation
        for sp, rate in [("LuxI", 0.03), ("AHL_int", 0.02), ("AHL_ext", 0.01),
                          ("LuxR", 0.01), ("LuxR_AHL", 0.015), ("GFP", 0.02),
                          ("RFP", 0.025), ("mRNA_LuxI", 0.1),
                          ("mRNA_LuxR", 0.1), ("mRNA_GFP", 0.1),
                          ("LasR", 0.01), ("LasI", 0.03)]:
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=rate),
            ))

        # Threshold activation and boundedness
        activation = Eventually(
            _pred_ge("GFP", 20.0), Interval(0.0, 500.0),
        )
        bounded = Always(
            STLAnd(_pred_le("GFP", 300.0), _pred_le("RFP", 300.0)),
            Interval(0.0, 500.0),
        )
        spec = STLAnd(activation, bounded)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.BOUNDED_GUARANTEE,
            difficulty=BenchmarkDifficulty.FRONTIER,
            name="quorum_sensing_extended",
            description="12-species quorum sensing with dual reporters",
            category="multicellular",
            tags=["quorum", "signaling", "reporter", "large"],
            metadata={
                "species_count": 12, "parameter_count": 14,
                "expected_time_s": 250,
            },
        )

    @staticmethod
    def repressilator_5node() -> BenchmarkCircuit:
        """Repressilator 5-node (5 species, 20 parameters).

        Five-gene ring oscillator where each gene represses the next.
        Uses the existing n_node_repressilator but exposed as a named
        benchmark for the 10-20 species scalability suite.
        """
        return BenchmarkSuite.n_node_repressilator(5)

    @staticmethod
    def large_feed_forward_network() -> BenchmarkCircuit:
        """Large feed-forward network (20 species, 40 parameters).

        Four-layer feed-forward network: 3 input nodes, 6 hidden-layer-1
        nodes, 6 hidden-layer-2 nodes, 3 output nodes, plus 2 global
        regulators.  Each layer feeds forward via Hill activation.
        """
        model = BioModel(name="large_feed_forward_network")

        inputs = [f"In{i}" for i in range(3)]
        hidden1 = [f"H1_{i}" for i in range(6)]
        hidden2 = [f"H2_{i}" for i in range(6)]
        outputs = [f"Out{i}" for i in range(3)]
        regulators = ["GlobReg1", "GlobReg2"]
        all_species = inputs + hidden1 + hidden2 + outputs + regulators

        for i, sp in enumerate(all_species):
            ic = 50.0 if sp in inputs else (10.0 if sp in regulators else 0.0)
            model.add_species(Species(sp, initial_concentration=ic,
                                       species_type=SpeciesType.PROTEIN))

        # Parameters: 2 per edge (Vmax, K) + degradation
        param_idx = 0
        edges = []
        # Input -> Hidden1
        for inp in inputs:
            for h in hidden1[:2]:
                edges.append((inp, h))
        # Hidden1 -> Hidden2
        for h1 in hidden1:
            for h2 in hidden2[:2]:
                edges.append((h1, h2))
        # Hidden2 -> Output
        for h2 in hidden2:
            for out in outputs[:1]:
                edges.append((h2, out))
        # Global regulators modulate hidden layers
        for reg in regulators:
            edges.append((reg, hidden1[0]))
            edges.append((reg, hidden2[0]))

        for src, tgt in edges:
            vmax = 4.0 + param_idx * 0.1
            K = 100.0 + param_idx * 5.0
            model.parameters.add(
                _hill_param(f"Vmax_{src}_{tgt}", vmax, 1.5, 10.0, "nM/min"))
            model.parameters.add(
                _hill_param(f"K_{src}_{tgt}", K, 40.0, 300.0))
            model.add_reaction(Reaction(
                f"act_{src}_{tgt}", [], [StoichiometryEntry(tgt)],
                HillActivation(Vmax=vmax, K=K, n=2.0),
            ))
            param_idx += 1

        # Degradation for all species
        for sp in all_species:
            rate = 0.03
            model.parameters.add(_rate_param(f"deg_{sp}", rate, 0.01, 0.06))
            model.add_reaction(Reaction(
                f"deg_{sp}", [StoichiometryEntry(sp)], [],
                LinearDegradation(rate=rate),
            ))

        # Spec: output eventually rises; all bounded
        liveness = Eventually(
            _pred_ge("Out0", 20.0), Interval(0.0, 600.0),
        )
        bounded = Always(
            _pred_le("Out0", 500.0), Interval(0.0, 600.0),
        )
        spec = STLAnd(liveness, bounded)

        return BenchmarkCircuit(
            model=model, specification=spec,
            expected_result=VerificationStatus.BOUNDED_GUARANTEE,
            difficulty=BenchmarkDifficulty.FRONTIER,
            name="large_feed_forward_network",
            description="20-species 4-layer feed-forward regulatory network",
            category="feed_forward",
            tags=["feed_forward", "large_scale", "layered", "large"],
            metadata={
                "species_count": 20, "parameter_count": 40,
                "expected_time_s": 600,
            },
        )

    # ----------------------------------------------------------------
    # Aggregate accessors
    # ----------------------------------------------------------------

    @classmethod
    def all_benchmarks(cls) -> List[BenchmarkCircuit]:
        """Return every built-in benchmark circuit."""
        benchmarks: List[BenchmarkCircuit] = [
            cls.toggle_switch(),
            cls.repressilator(),
            cls.nand_gate(),
            cls.nor_gate(),
        ]
        for lt in ["C1-I1", "C1-I2", "C1-I3", "C1-I4",
                    "C2-I1", "C2-I2", "C2-I3", "C2-I4"]:
            benchmarks.append(cls.feed_forward_loop(lt))
        for depth in range(2, 6):
            benchmarks.append(cls.cascade(depth))
        for n in [3, 5]:
            benchmarks.append(cls.n_node_repressilator(n))
        benchmarks.append(cls.random_grn())
        # Large-scale benchmarks (10+ species)
        benchmarks.append(cls.mapk_cascade())
        benchmarks.append(cls.quorum_sensing())
        benchmarks.append(cls.genetic_clock_reporter())
        benchmarks.append(cls.metabolic_toggle_feedback())
        benchmarks.append(cls.biosensor_amplification())
        # Additional 10-20 species benchmarks
        benchmarks.append(cls.mapk_3tier_cascade())
        benchmarks.append(cls.genetic_oscillator_reporter())
        benchmarks.append(cls.quorum_sensing_extended())
        benchmarks.append(cls.repressilator_5node())
        benchmarks.append(cls.large_feed_forward_network())
        return benchmarks

    @classmethod
    def by_difficulty(
        cls, difficulty: BenchmarkDifficulty,
    ) -> List[BenchmarkCircuit]:
        """Return benchmarks matching the given difficulty level."""
        return [b for b in cls.all_benchmarks()
                if b.difficulty == difficulty]

    @classmethod
    def by_category(cls, category: str) -> List[BenchmarkCircuit]:
        """Return benchmarks matching the given category string."""
        return [b for b in cls.all_benchmarks()
                if b.category == category]

    @classmethod
    def by_tags(cls, tags: Sequence[str]) -> List[BenchmarkCircuit]:
        """Return benchmarks whose tag set intersects *tags*."""
        tag_set = set(tags)
        return [b for b in cls.all_benchmarks()
                if tag_set & set(b.tags)]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Execute benchmark suites and collect :class:`BenchmarkResult` s.

    Parameters
    ----------
    suite:
        The benchmark suite to draw circuits from.
    config:
        CEGAR engine configuration applied to every run.
    timeout_per_benchmark:
        Wall-clock timeout in seconds for each individual benchmark.
    max_workers:
        Thread pool size for parallel execution (1 = sequential).
    """

    def __init__(
        self,
        suite: BenchmarkSuite,
        config: Optional[CEGARConfig] = None,
        timeout_per_benchmark: float = 600.0,
        max_workers: int = 1,
    ) -> None:
        self.suite = suite
        self.config = config or CEGARConfig(max_iterations=50,
                                            timeout=timeout_per_benchmark)
        self.timeout = timeout_per_benchmark
        self.max_workers = max_workers

    # -- Public interface -----------------------------------------------

    def run_all(self) -> List[BenchmarkResult]:
        """Run every benchmark in the suite."""
        return self._run_benchmarks(self.suite.all_benchmarks())

    def run_single(self, benchmark: BenchmarkCircuit) -> BenchmarkResult:
        """Run a single benchmark and return its result."""
        return self._execute_one(benchmark)

    def run_filtered(
        self,
        difficulty: Optional[BenchmarkDifficulty] = None,
        category: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> List[BenchmarkResult]:
        """Run a filtered subset of benchmarks."""
        candidates = self.suite.all_benchmarks()
        if difficulty is not None:
            candidates = [b for b in candidates
                          if b.difficulty == difficulty]
        if category is not None:
            candidates = [b for b in candidates
                          if b.category == category]
        if tags is not None:
            tag_set = set(tags)
            candidates = [b for b in candidates
                          if tag_set & set(b.tags)]
        return self._run_benchmarks(candidates)

    # -- Result aggregation ---------------------------------------------

    @staticmethod
    def aggregate_results(
        results: List[BenchmarkResult],
    ) -> Dict[str, Any]:
        """Compute summary statistics over a list of results.

        Returns a dictionary with keys: ``total``, ``passed``,
        ``failed``, ``errors``, ``pass_rate``, ``avg_time``,
        ``max_time``, ``by_difficulty``, ``by_category``.
        """
        total = len(results)
        if total == 0:
            return {"total": 0, "passed": 0, "failed": 0, "errors": 0,
                    "pass_rate": 0.0, "avg_time": 0.0, "max_time": 0.0,
                    "by_difficulty": {}, "by_category": {}}

        passed = sum(1 for r in results if r.correct)
        errors = sum(1 for r in results if r.error is not None)
        times = [r.wall_time for r in results]

        by_difficulty: Dict[str, Dict[str, Any]] = {}
        by_category: Dict[str, Dict[str, Any]] = {}

        for r in results:
            # group by difficulty
            d_key = r.benchmark.difficulty.name
            d_grp = by_difficulty.setdefault(
                d_key, {"total": 0, "passed": 0, "avg_time": 0.0})
            d_grp["total"] += 1
            d_grp["passed"] += int(r.correct)
            d_grp["avg_time"] += r.wall_time

            # group by category
            c_key = r.benchmark.category
            c_grp = by_category.setdefault(
                c_key, {"total": 0, "passed": 0, "avg_time": 0.0})
            c_grp["total"] += 1
            c_grp["passed"] += int(r.correct)
            c_grp["avg_time"] += r.wall_time

        for grp in list(by_difficulty.values()) + list(by_category.values()):
            if grp["total"]:
                grp["avg_time"] /= grp["total"]

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed - errors,
            "errors": errors,
            "pass_rate": passed / total if total else 0.0,
            "avg_time": sum(times) / total,
            "max_time": max(times),
            "by_difficulty": by_difficulty,
            "by_category": by_category,
        }

    @staticmethod
    def format_results_table(results: List[BenchmarkResult]) -> str:
        """Format benchmark results as an ASCII table."""
        header = (f"{'Benchmark':<30} {'Difficulty':<10} {'Expected':<12} "
                  f"{'Actual':<12} {'Time (s)':>10} {'Mem (MB)':>10} "
                  f"{'Status':>8}")
        sep = "-" * len(header)
        lines = [sep, header, sep]

        for r in results:
            actual_str = (r.actual_result.status.name
                          if r.actual_result else "ERROR")
            status = "PASS" if r.correct else ("ERR" if r.error else "FAIL")
            lines.append(
                f"{r.benchmark.name:<30} "
                f"{r.benchmark.difficulty.name:<10} "
                f"{r.benchmark.expected_result.name:<12} "
                f"{actual_str:<12} "
                f"{r.wall_time:>10.2f} "
                f"{r.peak_memory_mb:>10.1f} "
                f"{status:>8}"
            )

        lines.append(sep)
        summary = BenchmarkRunner.aggregate_results(results)
        lines.append(
            f"Total: {summary['total']}  "
            f"Passed: {summary['passed']}  "
            f"Failed: {summary['failed']}  "
            f"Errors: {summary['errors']}  "
            f"Rate: {summary['pass_rate']:.1%}  "
            f"Avg time: {summary['avg_time']:.2f}s"
        )
        return "\n".join(lines)

    # -- Internal execution ---------------------------------------------

    def _run_benchmarks(
        self, benchmarks: List[BenchmarkCircuit],
    ) -> List[BenchmarkResult]:
        """Execute a list of benchmarks, optionally in parallel."""
        if self.max_workers <= 1:
            return [self._execute_one(b) for b in benchmarks]

        results: List[BenchmarkResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._execute_one, b): b
                       for b in benchmarks}
            for future in futures:
                try:
                    results.append(future.result(timeout=self.timeout * 1.1))
                except FuturesTimeout:
                    bm = futures[future]
                    results.append(BenchmarkResult(
                        benchmark=bm, actual_result=None,
                        wall_time=self.timeout, peak_memory_mb=0.0,
                        correct=False, error="Timed out (executor)",
                    ))
                except Exception as exc:  # noqa: BLE001
                    bm = futures[future]
                    results.append(BenchmarkResult(
                        benchmark=bm, actual_result=None,
                        wall_time=0.0, peak_memory_mb=0.0,
                        correct=False, error=str(exc),
                    ))
        return results

    def _execute_one(self, benchmark: BenchmarkCircuit) -> BenchmarkResult:
        """Run a single benchmark with timeout and memory tracking."""
        logger.info("Running benchmark: %s", benchmark.name)
        t0 = time.monotonic()
        peak_mem = 0.0

        try:
            peak_mem = self._current_memory_mb()
            result = self._invoke_cegar(benchmark)
            elapsed = time.monotonic() - t0
            peak_mem = max(peak_mem, self._current_memory_mb()) - peak_mem

            correct = result.status == benchmark.expected_result
            return BenchmarkResult(
                benchmark=benchmark, actual_result=result,
                wall_time=elapsed, peak_memory_mb=max(0.0, peak_mem),
                correct=correct,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            logger.warning("Benchmark %s failed: %s", benchmark.name, exc)
            return BenchmarkResult(
                benchmark=benchmark, actual_result=None,
                wall_time=elapsed, peak_memory_mb=0.0,
                correct=False,
                error=f"{type(exc).__name__}: {exc}\n"
                      f"{traceback.format_exc()}",
            )

    def _invoke_cegar(
        self, benchmark: BenchmarkCircuit,
    ) -> VerificationResult:
        """Set up and run the CEGAR engine for a single benchmark.

        This is the integration point between the benchmark harness and
        the verification back-end.  Subclass :class:`BenchmarkRunner`
        and override this method to plug in alternative engines.
        """
        model = benchmark.model
        species_names = ([s.name for s in model.species]
                         if hasattr(model, "species") else [])
        bounds = {s: (0.0, 500.0) for s in species_names}

        engine = CEGAREngine(
            bounds=bounds,
            rhs={},
            property_expr=None,   # type: ignore[arg-type]
            property_name=benchmark.name,
            config=self.config,
        )
        return engine.verify()

    @staticmethod
    def _current_memory_mb() -> float:
        """Best-effort resident-set-size query (platform-dependent)."""
        try:
            import resource  # Unix only
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024.0  # macOS reports in bytes
        except ImportError:
            return 0.0
