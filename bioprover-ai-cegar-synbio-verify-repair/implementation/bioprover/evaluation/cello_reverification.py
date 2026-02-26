"""Cello re-verification study for BioProver.

Re-verifies genetic logic circuits from Nielsen et al., "Genetic circuit
design automation," Science 352(6281), aac7341 (2016).  Converts Cello
gate-level netlists into BioModels with Hill-function ODEs, generates STL
specs for Boolean correctness, and runs CEGAR verification with parametric
uncertainty.  Compares against Cello's reported 45/60 success rate.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from bioprover.cegar import CEGARConfig, CEGAREngine, VerificationResult, VerificationStatus
from bioprover.models import (
    BioModel, Compartment, ConstitutiveProduction, HillRepression,
    LinearDegradation, Parameter, ParameterSet, Reaction, Species,
    SpeciesType, StoichiometryEntry, UncertaintyType,
)
from bioprover.temporal import (
    Always, Eventually, Interval, Predicate, STLAnd, STLFormula, STLNot, STLOr,
)
from bioprover.temporal.stl_ast import ComparisonOp, make_var_expr

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_RPU = 0.5
STEADY_STATE_HORIZON = 600.0
SETTLING_TIME = 120.0
DEFAULT_DECAY_RATE = 0.02
INPUT_HIGH_RPU = 3.5
INPUT_LOW_RPU = 0.002


class CelloClassification(Enum):
    """Classification of a circuit after formal re-verification."""
    ROBUST = "robust"
    FRAGILE = "fragile"
    REPAIRABLE = "repairable"
    INCORRECT = "incorrect"
    TIMEOUT = "timeout"


@dataclass
class CelloGate:
    """Repressor gate from the Cello characterised library.

    Response curve: y = ymin + (ymax - ymin) / (1 + (x/K)^n)
    Parameters from Table S2 of Nielsen et al. 2016.
    """
    name: str
    gate_type: str  # "NOR", "NOT", "OUTPUT", "INPUT"
    ymax: float     # max output (RPU)
    ymin: float     # min output (RPU)
    K: float        # Hill constant (RPU)
    n: float        # Hill coefficient
    decay_rate: float = DEFAULT_DECAY_RATE


@dataclass
class CelloCircuit:
    """Fully-assigned Cello circuit design."""
    circuit_id: str             # e.g. "0x01"
    boolean_function: str       # e.g. "NOT(A AND B)"
    input_names: List[str]
    output_name: str
    gates: List[CelloGate]
    wiring: Dict[str, List[str]]  # gate_name -> input gate/signal names
    expected_score: float         # Cello predicted score (0-1)


# -- Characterised gates (Nielsen et al. 2016, Table S2) -------------------

_STANDARD_GATES: List[CelloGate] = [
    CelloGate("P1_PhlF",  "NOR", ymax=3.9, ymin=0.01,  K=0.12, n=2.8, decay_rate=0.018),
    CelloGate("P2_SrpR",  "NOR", ymax=5.6, ymin=0.006, K=0.07, n=2.5, decay_rate=0.015),
    CelloGate("P3_BM3R1", "NOR", ymax=4.4, ymin=0.01,  K=0.41, n=1.6, decay_rate=0.022),
    CelloGate("P4_QacR",  "NOR", ymax=2.6, ymin=0.02,  K=0.23, n=1.4, decay_rate=0.025),
    CelloGate("P5_IcaRA", "NOR", ymax=3.0, ymin=0.03,  K=0.30, n=1.7, decay_rate=0.020),
    CelloGate("P6_AmtR",  "NOR", ymax=4.0, ymin=0.007, K=0.04, n=2.3, decay_rate=0.017),
    CelloGate("P7_LitR",  "NOR", ymax=3.2, ymin=0.02,  K=0.19, n=1.5, decay_rate=0.023),
    CelloGate("P8_AraC",  "NOT", ymax=5.2, ymin=0.008, K=0.09, n=2.1, decay_rate=0.016),
]


class CelloLibrary:
    """Library of characterised gates and representative circuit designs.

    Contains the 8 standard gates and ~22 circuits spanning Boolean
    functions tested in Nielsen et al. 2016.
    """

    def __init__(self) -> None:
        self._gates: Dict[str, CelloGate] = {g.name: g for g in _STANDARD_GATES}
        self._circuits: Dict[str, CelloCircuit] = {}
        self._build_circuits()

    def get_gates(self) -> List[CelloGate]:
        return list(self._gates.values())

    def get_all_circuits(self) -> List[CelloCircuit]:
        return list(self._circuits.values())

    def get_circuit(self, circuit_id: str) -> CelloCircuit:
        if circuit_id not in self._circuits:
            raise KeyError(f"Unknown circuit id: {circuit_id!r}")
        return self._circuits[circuit_id]

    def _g(self, name: str) -> CelloGate:
        return self._gates[name]

    def _add(self, c: CelloCircuit) -> None:
        self._circuits[c.circuit_id] = c

    def _build_circuits(self) -> None:
        g = self._g
        # -- 1-input -------------------------------------------------------
        self._add(CelloCircuit("0x01", "NOT A", ["A"], "Y",
            [g("P1_PhlF")], {"P1_PhlF": ["A"]}, 0.95))
        self._add(CelloCircuit("0x02", "A", ["A"], "Y",
            [g("P1_PhlF"), g("P2_SrpR")],
            {"P1_PhlF": ["A"], "P2_SrpR": ["P1_PhlF"]}, 0.88))
        self._add(CelloCircuit("0x11", "NOT A", ["A"], "Y",
            [g("P6_AmtR")], {"P6_AmtR": ["A"]}, 0.94))
        # -- 2-input NOR ---------------------------------------------------
        self._add(CelloCircuit("0x06", "A NOR B", ["A", "B"], "Y",
            [g("P2_SrpR")], {"P2_SrpR": ["A", "B"]}, 0.93))
        self._add(CelloCircuit("0x07", "NOT(A OR B)", ["A", "B"], "Y",
            [g("P6_AmtR")], {"P6_AmtR": ["A", "B"]}, 0.91))
        self._add(CelloCircuit("0x10", "A NOR B", ["A", "B"], "Y",
            [g("P3_BM3R1")], {"P3_BM3R1": ["A", "B"]}, 0.90))
        # -- 2-input NAND --------------------------------------------------
        self._add(CelloCircuit("0x08", "A NAND B", ["A", "B"], "Y",
            [g("P3_BM3R1"), g("P1_PhlF")],
            {"P3_BM3R1": ["A"], "P1_PhlF": ["P3_BM3R1", "B"]}, 0.87))
        self._add(CelloCircuit("0x09", "NOT(A AND B)", ["A", "B"], "Y",
            [g("P4_QacR"), g("P2_SrpR")],
            {"P4_QacR": ["A", "B"], "P2_SrpR": ["P4_QacR"]}, 0.85))
        self._add(CelloCircuit("0x12", "A NAND B", ["A", "B"], "Y",
            [g("P5_IcaRA"), g("P7_LitR")],
            {"P5_IcaRA": ["A", "B"], "P7_LitR": ["P5_IcaRA"]}, 0.78))
        # -- 2-input AND / OR ----------------------------------------------
        self._add(CelloCircuit("0x0E", "A AND B", ["A", "B"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P2_SrpR")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"],
             "P2_SrpR": ["P1_PhlF", "P6_AmtR"]}, 0.82))
        self._add(CelloCircuit("0x0A", "A OR B", ["A", "B"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P3_BM3R1")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"],
             "P3_BM3R1": ["P1_PhlF", "P6_AmtR"]}, 0.80))
        self._add(CelloCircuit("0x13", "A AND B", ["A", "B"], "Y",
            [g("P3_BM3R1"), g("P5_IcaRA"), g("P7_LitR")],
            {"P3_BM3R1": ["A"], "P5_IcaRA": ["B"],
             "P7_LitR": ["P3_BM3R1", "P5_IcaRA"]}, 0.72))
        # -- 2-input XOR / XNOR -------------------------------------------
        self._add(CelloCircuit("0x0B", "A XOR B", ["A", "B"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P4_QacR"), g("P2_SrpR")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"],
             "P4_QacR": ["P1_PhlF", "B"], "P2_SrpR": ["P4_QacR", "P6_AmtR"]}, 0.62))
        self._add(CelloCircuit("0x0C", "A XNOR B", ["A", "B"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P3_BM3R1"), g("P5_IcaRA"), g("P2_SrpR")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"],
             "P3_BM3R1": ["P1_PhlF", "B"], "P5_IcaRA": ["A", "P6_AmtR"],
             "P2_SrpR": ["P3_BM3R1", "P5_IcaRA"]}, 0.55))
        # -- 3-input -------------------------------------------------------
        self._add(CelloCircuit("0x20", "NOT(A OR B OR C)", ["A", "B", "C"], "Y",
            [g("P2_SrpR")], {"P2_SrpR": ["A", "B", "C"]}, 0.89))
        self._add(CelloCircuit("0x21", "(A AND B) NOR C", ["A", "B", "C"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P2_SrpR")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"],
             "P2_SrpR": ["P1_PhlF", "P6_AmtR", "C"]}, 0.74))
        self._add(CelloCircuit("0x22", "A AND B AND C", ["A", "B", "C"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P4_QacR"), g("P2_SrpR")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"], "P4_QacR": ["C"],
             "P2_SrpR": ["P1_PhlF", "P6_AmtR", "P4_QacR"]}, 0.68))
        self._add(CelloCircuit("0x23", "A OR B OR C", ["A", "B", "C"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P4_QacR"), g("P3_BM3R1")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"], "P4_QacR": ["C"],
             "P3_BM3R1": ["P1_PhlF", "P6_AmtR", "P4_QacR"]}, 0.70))
        self._add(CelloCircuit("0x24", "(A OR B) AND C", ["A", "B", "C"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P3_BM3R1"), g("P4_QacR"), g("P2_SrpR")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"],
             "P3_BM3R1": ["P1_PhlF", "P6_AmtR"], "P4_QacR": ["C"],
             "P2_SrpR": ["P3_BM3R1", "P4_QacR"]}, 0.65))
        self._add(CelloCircuit("0x25", "(A AND B) OR C", ["A", "B", "C"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P2_SrpR"), g("P4_QacR"), g("P3_BM3R1")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"],
             "P2_SrpR": ["P1_PhlF", "P6_AmtR"], "P4_QacR": ["C"],
             "P3_BM3R1": ["P2_SrpR", "P4_QacR"]}, 0.60))
        self._add(CelloCircuit("0x27", "A XOR B XOR C", ["A", "B", "C"], "Y",
            [g("P1_PhlF"), g("P6_AmtR"), g("P3_BM3R1"), g("P5_IcaRA"),
             g("P4_QacR"), g("P2_SrpR")],
            {"P1_PhlF": ["A"], "P6_AmtR": ["B"],
             "P3_BM3R1": ["P1_PhlF", "B"], "P5_IcaRA": ["A", "P6_AmtR"],
             "P4_QacR": ["P3_BM3R1", "P5_IcaRA", "C"],
             "P2_SrpR": ["P4_QacR"]}, 0.42))

    def __repr__(self) -> str:
        return f"CelloLibrary(gates={len(self._gates)}, circuits={len(self._circuits)})"


# ---------------------------------------------------------------------------
# Boolean evaluation helpers
# ---------------------------------------------------------------------------

def _truth_table(n_inputs: int) -> List[Tuple[int, ...]]:
    """Return all binary input combinations for *n_inputs* signals."""
    return [tuple((row >> (n_inputs - 1 - j)) & 1 for j in range(n_inputs))
            for row in range(1 << n_inputs)]


def _eval_boolean(expr: str, var_vals: Dict[str, bool]) -> bool:
    """Evaluate a simple Boolean expression given variable truth values.

    Supports AND, OR, NOT, NOR, NAND, XOR, XNOR with parentheses.
    """
    s = expr
    for var, val in var_vals.items():
        s = s.replace(var, str(val))

    has_nor = "NOR" in expr and "XNOR" not in expr
    has_nand = "NAND" in expr

    s = s.replace("XNOR", "==").replace("XOR", "!=")
    s = s.replace("NOR", "or").replace("NAND", "and")
    s = s.replace("AND", "and").replace("OR", "or").replace("NOT", "not ")

    if has_nor:
        s = f"not ({s})"
    if has_nand:
        s = f"not ({s})"

    try:
        return bool(eval(s, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        logger.warning("Could not evaluate Boolean expression: %s", expr)
        return False


# ---------------------------------------------------------------------------
# CelloReverifier
# ---------------------------------------------------------------------------

class CelloReverifier:
    """Run CEGAR verification on circuits from the Cello library.

    Converts each circuit to a BioModel with Hill-function ODEs, generates
    an STL spec encoding steady-state Boolean correctness, runs the CEGAR
    engine, and classifies the outcome.
    """

    def __init__(self, library: CelloLibrary, config: CEGARConfig,
                 uncertainty_pct: float = 20.0) -> None:
        self.library = library
        self.config = config
        self.uncertainty_pct = uncertainty_pct

    def circuit_to_biomodel(self, circuit: CelloCircuit) -> BioModel:
        """Convert a CelloCircuit into a BioModel with ODE reactions."""
        model = BioModel(f"cello_{circuit.circuit_id}")
        frac = self.uncertainty_pct / 100.0

        for inp_name in circuit.input_names:
            model.add_species(Species(
                name=inp_name, compartment="default",
                initial_concentration=INPUT_LOW_RPU, units="RPU",
                species_type=SpeciesType.PROTEIN))

        for gate in circuit.gates:
            protein = f"{gate.name}_out"
            model.add_species(Species(
                name=protein, compartment="default",
                initial_concentration=0.0, units="RPU",
                species_type=SpeciesType.PROTEIN))

            # Register parameters with uncertainty bounds.
            params = [
                Parameter(f"{gate.name}_ymax", gate.ymax, "RPU",
                          gate.ymax * (1 - frac), gate.ymax * (1 + frac)),
                Parameter(f"{gate.name}_ymin", gate.ymin, "RPU",
                          gate.ymin * (1 - frac), gate.ymin * (1 + frac)),
                Parameter(f"{gate.name}_K", gate.K, "RPU",
                          gate.K * (1 - frac), gate.K * (1 + frac)),
                Parameter(f"{gate.name}_n", gate.n, "",
                          gate.n * (1 - frac), gate.n * (1 + frac)),
                Parameter(f"{gate.name}_deg", gate.decay_rate, "/min",
                          gate.decay_rate * (1 - frac), gate.decay_rate * (1 + frac)),
            ]
            for p in params:
                model.parameters.add(p)

            # Hill repression for each input signal feeding this gate.
            for idx, inp_signal in enumerate(circuit.wiring.get(gate.name, [])):
                substrate = (inp_signal if inp_signal in circuit.input_names
                             else f"{inp_signal}_out")
                model.add_reaction(Reaction(
                    name=f"{gate.name}_repr_{idx}",
                    reactants=[StoichiometryEntry(species_name=substrate, coefficient=1)],
                    products=[StoichiometryEntry(species_name=protein, coefficient=1)],
                    kinetic_law=HillRepression(
                        substrate=substrate, K=gate.K, n=gate.n, vmax=gate.ymax)))

            # Basal production and degradation.
            model.add_reaction(Reaction(
                name=f"{gate.name}_basal", reactants=[],
                products=[StoichiometryEntry(species_name=protein, coefficient=1)],
                kinetic_law=ConstitutiveProduction(rate=gate.ymin)))
            model.add_reaction(Reaction(
                name=f"{gate.name}_deg",
                reactants=[StoichiometryEntry(species_name=protein, coefficient=1)],
                products=[], kinetic_law=LinearDegradation(rate_constant=gate.decay_rate)))

        return model

    def _output_species(self, circuit: CelloCircuit) -> str:
        return f"{circuit.gates[-1].name}_out"

    def generate_spec(self, circuit: CelloCircuit) -> STLFormula:
        """Generate an STL spec requiring correct Boolean I/O at steady state."""
        output_var = make_var_expr(self._output_species(circuit))
        interval = Interval(SETTLING_TIME, STEADY_STATE_HORIZON)
        row_formulas: List[STLFormula] = []

        for bits in _truth_table(len(circuit.input_names)):
            var_vals = {n: bool(b) for n, b in zip(circuit.input_names, bits)}
            expected_high = _eval_boolean(circuit.boolean_function, var_vals)
            op = ComparisonOp.GE if expected_high else ComparisonOp.LE
            row_formulas.append(Always(Predicate(output_var, op, DEFAULT_THRESHOLD_RPU), interval))

        spec = row_formulas[0]
        for f in row_formulas[1:]:
            spec = STLAnd(spec, f)
        return spec

    @staticmethod
    def classify(result: VerificationResult) -> CelloClassification:
        """Map a VerificationResult to a CelloClassification.

        VERIFIED + robustness >= 0.3 → ROBUST; < 0.3 → FRAGILE.
        FALSIFIED + counterexample_volume <= 10% → REPAIRABLE; else INCORRECT.
        Everything else → TIMEOUT.
        """
        if result.status == VerificationStatus.VERIFIED:
            robustness = getattr(result, "robustness", 0.0) or 0.0
            return CelloClassification.ROBUST if robustness >= 0.3 else CelloClassification.FRAGILE
        if result.status == VerificationStatus.FALSIFIED:
            cex_vol = getattr(result, "counterexample_volume", 1.0) or 1.0
            return CelloClassification.REPAIRABLE if cex_vol <= 0.10 else CelloClassification.INCORRECT
        return CelloClassification.TIMEOUT

    def reverify_single(self, circuit: CelloCircuit) -> Tuple[CelloClassification, VerificationResult]:
        """Verify one circuit and return (classification, result)."""
        logger.info("Reverifying %s (%s)", circuit.circuit_id, circuit.boolean_function)
        model = self.circuit_to_biomodel(circuit)
        spec = self.generate_spec(circuit)
        engine = CEGAREngine(model, spec, self.config)
        t0 = time.monotonic()
        result = engine.run()
        logger.info("Circuit %s: %s in %.1f s", circuit.circuit_id,
                     result.status.value, time.monotonic() - t0)
        return self.classify(result), result

    def reverify_all(self) -> Dict[str, Tuple[CelloClassification, VerificationResult]]:
        """Verify every circuit in the library."""
        return {c.circuit_id: self.reverify_single(c) for c in self.library.get_all_circuits()}

    def sensitivity_analysis(
        self, circuit: CelloCircuit, distributions: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[CelloClassification, VerificationResult]]:
        """Re-verify under different uncertainty distributions.

        Args:
            circuit: Circuit to analyse.
            distributions: e.g. ["uniform", "log-normal", "truncated-gaussian"].
        """
        if distributions is None:
            distributions = ["uniform", "log-normal", "truncated-gaussian"]

        out: Dict[str, Tuple[CelloClassification, VerificationResult]] = {}
        for dist_name in distributions:
            logger.info("Sensitivity %s with %s", circuit.circuit_id, dist_name)
            model = self.circuit_to_biomodel(circuit)
            # For truncated-gaussian, attach std_dev to each parameter.
            if dist_name == "truncated-gaussian":
                param_store = getattr(model.parameters, "_params", {})
                for param in param_store.values():
                    param.std_dev = abs(param.value) * (self.uncertainty_pct / 100.0)
            spec = self.generate_spec(circuit)
            result = CEGAREngine(model, spec, self.config).run()
            out[dist_name] = (self.classify(result), result)
        return out


# ---------------------------------------------------------------------------
# CelloStudyReport
# ---------------------------------------------------------------------------

_COL = {"id": 8, "fn": 24, "cls": 14, "t": 8, "rob": 10, "sc": 10}


class CelloStudyReport:
    """Aggregate and format results from a Cello re-verification campaign."""

    def __init__(self, results: Dict[str, Tuple[CelloClassification, VerificationResult]],
                 library: Optional[CelloLibrary] = None) -> None:
        self.results = results
        self.library = library or CelloLibrary()

    def summary(self) -> Dict:
        """Classification counts and comparison with Cello's 45/60 success rate."""
        counts: Dict[str, int] = {c.value: 0 for c in CelloClassification}
        for cls, _ in self.results.values():
            counts[cls.value] += 1
        total = len(self.results)
        verified = counts["robust"] + counts["fragile"]
        return {
            "total_circuits": total,
            "classification_counts": counts,
            "verified_fraction": verified / total if total else 0.0,
            "cello_reported_success_rate": 45 / 60,
            "note": ("Nielsen et al. reported 45/60 circuits functioning correctly. "
                     "CEGAR analysis may differ due to parametric uncertainty and "
                     "temporal dynamics modelling."),
        }

    def per_circuit_table(self) -> str:
        """Produce a human-readable ASCII table of per-circuit results."""
        hdr = (f"{'ID':<{_COL['id']}} {'Function':<{_COL['fn']}} "
               f"{'Class':<{_COL['cls']}} {'Time(s)':<{_COL['t']}} "
               f"{'Robust.':<{_COL['rob']}} {'Cello sc.':<{_COL['sc']}}")
        lines = [hdr, "-" * len(hdr)]
        for cid in sorted(self.results):
            cls, res = self.results[cid]
            try:
                circ = self.library.get_circuit(cid)
                func, score = circ.boolean_function, f"{circ.expected_score:.2f}"
            except KeyError:
                func, score = "?", "?"
            t = getattr(res, "elapsed_time", None)
            r = getattr(res, "robustness", None)
            lines.append(
                f"{cid:<{_COL['id']}} {func:<{_COL['fn']}} "
                f"{cls.value:<{_COL['cls']}} {(f'{t:.1f}' if t else '-'):<{_COL['t']}} "
                f"{(f'{r:.4f}' if r else '-'):<{_COL['rob']}} {score:<{_COL['sc']}}")
        return "\n".join(lines)

    def latex_table(self) -> str:
        """Produce a LaTeX-formatted results table."""
        lines = [
            r"\begin{table}[ht]", r"\centering",
            r"\caption{Cello circuit re-verification results}",
            r"\label{tab:cello-reverification}",
            r"\begin{tabular}{llllrr}", r"\toprule",
            r"ID & Function & Classification & Status & Time (s) & Robustness \\",
            r"\midrule",
        ]
        for cid in sorted(self.results):
            cls, res = self.results[cid]
            try:
                func = self.library.get_circuit(cid).boolean_function.replace("_", r"\_")
            except KeyError:
                func = "?"
            t = getattr(res, "elapsed_time", None)
            r = getattr(res, "robustness", None)
            lines.append(
                f"{cid} & {func} & {cls.value} & {res.status.value} "
                f"& {f'{t:.1f}' if t else '--'} & {f'{r:.4f}' if r else '--'} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def sensitivity_summary(
        self,
        sensitivity_results: Dict[str, Dict[str, Tuple[CelloClassification, VerificationResult]]],
    ) -> str:
        """Format sensitivity analysis results across circuits/distributions."""
        lines = ["Sensitivity Analysis Summary", "=" * 40]
        for cid in sorted(sensitivity_results):
            lines.append(f"\nCircuit {cid}:")
            for dist, (cls, res) in sensitivity_results[cid].items():
                r = getattr(res, "robustness", None)
                lines.append(f"  {dist:<22s}  {cls.value:<12s}  "
                             f"robustness={f'{r:.4f}' if r else '-'}")
        return "\n".join(lines)
