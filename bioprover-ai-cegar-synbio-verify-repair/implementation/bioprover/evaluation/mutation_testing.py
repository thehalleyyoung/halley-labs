"""Mutation testing for BioProver soundness validation.

Generates mutants of biological models/specifications and checks whether the
CEGAR verification pipeline detects introduced faults.  A high mutation score
indicates the verifier is sensitive to biologically meaningful changes.
"""
from __future__ import annotations
import copy, itertools, logging, time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from bioprover.cegar import CEGARConfig, CEGAREngine, VerificationResult, VerificationStatus
from bioprover.models import (
    BioModel, HillActivation, HillRepression, KineticLaw, LinearDegradation,
    MassAction, MichaelisMenten, ConstitutiveProduction, Parameter, ParameterSet,
    Reaction, Species, SpeciesType, StoichiometryEntry, UncertaintyType,
)
from bioprover.temporal import STLFormula, STLNot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & data-classes
# ---------------------------------------------------------------------------

class MutantStatus(Enum):
    """Outcome of verifying a single mutant."""
    KILLED = auto()
    SURVIVED = auto()
    EQUIVALENT = auto()
    ERROR = auto()
    TIMEOUT = auto()

@dataclass
class Mutant:
    """A model (or spec) variant produced by a mutation operator."""
    original_model: BioModel
    mutated_model: BioModel
    operator_name: str
    mutation_description: str
    mutation_location: str
    order: int = 1

@dataclass
class MutationReport:
    """Aggregated results of a mutation-testing campaign."""
    total_mutants: int = 0
    killed: int = 0
    survived: int = 0
    equivalent: int = 0
    errors: int = 0
    timeouts: int = 0
    details: List[Tuple[Mutant, MutantStatus]] = field(default_factory=list)
    operator_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def mutation_score(self) -> float:
        denom = self.total_mutants - self.equivalent
        return self.killed / denom if denom > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Mutation score: {self.mutation_score:.1%}  "
            f"({self.killed} killed / {self.total_mutants - self.equivalent} non-equivalent)",
            f"  Total: {self.total_mutants}  Killed: {self.killed}  "
            f"Survived: {self.survived}  Equivalent: {self.equivalent}  "
            f"Errors: {self.errors}  Timeouts: {self.timeouts}",
        ]
        if self.operator_scores:
            lines.append("  Per-operator scores:")
            for op, score in sorted(self.operator_scores.items()):
                lines.append(f"    {op:30s} {score:.1%}")
        return "\n".join(lines)

    def detailed_table(self) -> str:
        hdr = f"{'#':>4}  {'Operator':20s}  {'Location':25s}  {'Status':12s}  Description"
        sep = "-" * len(hdr)
        rows = [sep, hdr, sep]
        for i, (m, s) in enumerate(self.details, 1):
            rows.append(f"{i:4d}  {m.operator_name:20s}  {m.mutation_location:25s}  "
                        f"{s.name:12s}  {m.mutation_description}")
        rows.append(sep)
        return "\n".join(rows)

    def latex_table(self) -> str:
        op_counts: Dict[str, Dict[str, int]] = {}
        for m, s in self.details:
            e = op_counts.setdefault(m.operator_name, {"total": 0, "killed": 0})
            e["total"] += 1
            if s == MutantStatus.KILLED:
                e["killed"] += 1
        rows = []
        for op, c in sorted(op_counts.items()):
            sc = c["killed"] / c["total"] if c["total"] else 0.0
            rows.append(f"{op} & {c['total']} & {c['killed']} & {sc:.1%} \\\\")
        return "\n".join([
            r"\begin{table}[t]", r"\centering",
            r"\caption{Mutation testing results}", r"\label{tab:mutation}",
            r"\begin{tabular}{lrrr}", r"\toprule",
            r"Operator & Mutants & Killed & Score \\", r"\midrule",
            *rows, r"\midrule",
            f"Total & {self.total_mutants} & {self.killed} & "
            f"{self.mutation_score:.1%} \\\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ])

# ---------------------------------------------------------------------------
# Abstract mutation operator
# ---------------------------------------------------------------------------

class MutationOperator(ABC):
    """Base class for all mutation operators."""
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def description(self) -> str: ...
    @abstractmethod
    def generate(self, model: BioModel, rng: np.random.Generator) -> List[Mutant]: ...
    @abstractmethod
    def apply_single(self, model: BioModel, target: str,
                     rng: np.random.Generator) -> Optional[Mutant]: ...

# ---------------------------------------------------------------------------
# Concrete operators
# ---------------------------------------------------------------------------

class ParameterPerturbation(MutationOperator):
    """Multiply each parameter value (and bounds) by fixed factors."""
    def __init__(self, perturbation_factors: Optional[List[float]] = None) -> None:
        self._factors = perturbation_factors or [0.1, 0.5, 2.0, 10.0]

    @property
    def name(self) -> str: return "ParameterPerturbation"
    @property
    def description(self) -> str:
        return f"Scale each parameter by factors {self._factors}."

    def generate(self, model: BioModel, rng: np.random.Generator) -> List[Mutant]:
        out: List[Mutant] = []
        for pname in model.parameters.names:
            for f in self._factors:
                m = self.apply_single(model, f"{pname}:{f}", rng)
                if m is not None:
                    out.append(m)
        return out

    def apply_single(self, model: BioModel, target: str,
                     rng: np.random.Generator) -> Optional[Mutant]:
        parts = target.split(":")
        if len(parts) != 2:
            return None
        pname, factor = parts[0], float(parts[1])
        try:
            model.parameters.get(pname)
        except (KeyError, ValueError):
            return None
        clone = copy.deepcopy(model)
        p = clone.parameters.get(pname)
        p.value *= factor
        if p.lower_bound is not None:
            p.lower_bound *= factor
        if p.upper_bound is not None:
            p.upper_bound *= factor
        return Mutant(model, clone, self.name,
                      f"Parameter {pname} *= {factor}", pname)


class ReactionRemoval(MutationOperator):
    """Remove one reaction at a time."""
    @property
    def name(self) -> str: return "ReactionRemoval"
    @property
    def description(self) -> str: return "Delete a single reaction."

    def generate(self, model: BioModel, rng: np.random.Generator) -> List[Mutant]:
        if len(model.reactions) <= 1:
            return []
        return [m for rxn in model.reactions
                if (m := self.apply_single(model, rxn.name, rng)) is not None]

    def apply_single(self, model: BioModel, target: str,
                     rng: np.random.Generator) -> Optional[Mutant]:
        if len(model.reactions) <= 1:
            return None
        clone = copy.deepcopy(model)
        clone._reactions = {k: v for k, v in clone._reactions.items() if k != target}
        if not clone._reactions:
            return None
        return Mutant(model, clone, self.name,
                      f"Removed reaction '{target}'", target)


class SpeciesRemoval(MutationOperator):
    """Remove a non-input species and all reactions that reference it."""
    @property
    def name(self) -> str: return "SpeciesRemoval"
    @property
    def description(self) -> str: return "Remove a species and dependent reactions."

    def generate(self, model: BioModel, rng: np.random.Generator) -> List[Mutant]:
        out: List[Mutant] = []
        for sp in model.species:
            if sp.species_type == SpeciesType.PROMOTER:
                continue
            m = self.apply_single(model, sp.name, rng)
            if m is not None:
                out.append(m)
        return out

    def apply_single(self, model: BioModel, target: str,
                     rng: np.random.Generator) -> Optional[Mutant]:
        if target not in model.species_names:
            return None
        clone = copy.deepcopy(model)
        clone._species = {k: v for k, v in clone._species.items() if k != target}
        clone._reactions = {k: v for k, v in clone._reactions.items()
                            if target not in v.species_involved}
        if not clone._species or not clone._reactions:
            return None
        return Mutant(model, clone, self.name,
                      f"Removed species '{target}' and dependent reactions", target)


class KineticLawChange(MutationOperator):
    """Swap kinetic law type for a reaction (activation↔repression, etc.)."""
    @property
    def name(self) -> str: return "KineticLawChange"
    @property
    def description(self) -> str:
        return "Swap activation/repression or change mass-action order."

    def generate(self, model: BioModel, rng: np.random.Generator) -> List[Mutant]:
        return [m for rxn in model.reactions
                if (m := self.apply_single(model, rxn.name, rng)) is not None]

    def apply_single(self, model: BioModel, target: str,
                     rng: np.random.Generator) -> Optional[Mutant]:
        clone = copy.deepcopy(model)
        if target not in clone._reactions:
            return None
        rxn = clone._reactions[target]
        law = rxn.kinetic_law
        if isinstance(law, HillActivation):
            new_law = HillRepression(Vmax=law.Vmax, K=law.K, n=law.n)
            new_law.repressor_name = law.activator_name
            rxn.kinetic_law = new_law
            desc = f"HillActivation → HillRepression in '{target}'"
        elif isinstance(law, HillRepression):
            new_law = HillActivation(Vmax=law.Vmax, K=law.K, n=law.n)
            new_law.activator_name = law.repressor_name
            rxn.kinetic_law = new_law
            desc = f"HillRepression → HillActivation in '{target}'"
        elif isinstance(law, MassAction):
            rxn.kinetic_law = MassAction(k_forward=law.k_forward, k_reverse=law.k_reverse)
            desc = f"MassAction order change in '{target}'"
        else:
            return None
        return Mutant(model, clone, self.name, desc, target)


class StoichiometryChange(MutationOperator):
    """Modify stoichiometric coefficients by ±1."""
    @property
    def name(self) -> str: return "StoichiometryChange"
    @property
    def description(self) -> str: return "Perturb stoichiometric coefficients."

    def generate(self, model: BioModel, rng: np.random.Generator) -> List[Mutant]:
        out: List[Mutant] = []
        for rxn in model.reactions:
            for i in range(len(rxn.reactants)):
                m = self.apply_single(model, f"{rxn.name}:reactant:{i}", rng)
                if m is not None:
                    out.append(m)
            for i in range(len(rxn.products)):
                m = self.apply_single(model, f"{rxn.name}:product:{i}", rng)
                if m is not None:
                    out.append(m)
        return out

    def apply_single(self, model: BioModel, target: str,
                     rng: np.random.Generator) -> Optional[Mutant]:
        parts = target.split(":")
        if len(parts) != 3:
            return None
        rxn_name, role, idx = parts[0], parts[1], int(parts[2])
        clone = copy.deepcopy(model)
        if rxn_name not in clone._reactions:
            return None
        rxn = clone._reactions[rxn_name]
        entries = rxn.reactants if role == "reactant" else rxn.products
        if idx >= len(entries):
            return None
        old = entries[idx]
        new_coeff = max(1, old.coefficient + int(rng.choice([-1, 1])))
        entries[idx] = StoichiometryEntry(species_name=old.species_name,
                                          coefficient=new_coeff)
        return Mutant(model, clone, self.name,
                      f"{role.capitalize()} '{old.species_name}' coeff "
                      f"{old.coefficient} → {new_coeff} in '{rxn_name}'",
                      f"{rxn_name}/{old.species_name}")


class PropertyNegation(MutationOperator):
    """Negate the STL specification to validate soundness.

    Unlike other operators this mutates the *property*, not the model.
    """
    def __init__(self) -> None:
        self._spec: Optional[STLFormula] = None

    @property
    def name(self) -> str: return "PropertyNegation"
    @property
    def description(self) -> str: return "Negate the STL specification."

    def set_spec(self, spec: STLFormula) -> None:
        self._spec = spec

    def generate(self, model: BioModel, rng: np.random.Generator) -> List[Mutant]:
        if self._spec is None:
            return []
        mutant = Mutant(model, copy.deepcopy(model), self.name,
                        "Negated specification via STLNot", "specification")
        mutant._negated_spec = STLNot(self._spec)  # type: ignore[attr-defined]
        return [mutant]

    def apply_single(self, model: BioModel, target: str,
                     rng: np.random.Generator) -> Optional[Mutant]:
        ms = self.generate(model, rng)
        return ms[0] if ms else None

# ---------------------------------------------------------------------------
# Mutation test runner
# ---------------------------------------------------------------------------

class MutationTestRunner:
    """Orchestrates mutant generation, verification, and reporting."""

    def __init__(self, model: BioModel, spec: STLFormula,
                 operators: List[MutationOperator], config: CEGARConfig,
                 timeout: float = 300.0,
                 verify_fn: Optional[Callable[[BioModel, STLFormula],
                                              VerificationResult]] = None,
                 ) -> None:
        self.model = model
        self.spec = spec
        self.operators = operators
        self.config = config
        self.timeout = timeout
        self._rng = np.random.default_rng(42)
        self._verify_fn = verify_fn

    def generate_mutants(self, max_per_operator: int = 50) -> List[Mutant]:
        """Collect mutants from every registered operator."""
        all_mutants: List[Mutant] = []
        for op in self.operators:
            if isinstance(op, PropertyNegation):
                op.set_spec(self.spec)
            mutants = op.generate(self.model, self._rng)
            if len(mutants) > max_per_operator:
                idx = self._rng.choice(len(mutants), max_per_operator, replace=False)
                mutants = [mutants[i] for i in sorted(idx)]
            all_mutants.extend(mutants)
            logger.info("Operator %s produced %d mutant(s)", op.name, len(mutants))
        return all_mutants

    def run_single(self, mutant: Mutant,
                   original_result: VerificationResult,
                   ) -> Tuple[MutantStatus, Optional[VerificationResult]]:
        """Verify *mutant* and compare outcome to *original_result*."""
        spec = getattr(mutant, "_negated_spec", self.spec)
        try:
            t0 = time.monotonic()
            result = (self._verify_fn(mutant.mutated_model, spec)
                      if self._verify_fn else self._default_verify(mutant.mutated_model, spec))
            if time.monotonic() - t0 > self.timeout:
                return MutantStatus.TIMEOUT, result
            if result.status != original_result.status:
                return MutantStatus.KILLED, result
            if self._detect_equivalent(mutant):
                return MutantStatus.EQUIVALENT, result
            return MutantStatus.SURVIVED, result
        except TimeoutError:
            return MutantStatus.TIMEOUT, None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error verifying mutant: %s", exc)
            return MutantStatus.ERROR, None

    def run_all(self, original_result: VerificationResult) -> MutationReport:
        """Run the full mutation-testing campaign."""
        mutants = self.generate_mutants()
        report = MutationReport(total_mutants=len(mutants))
        status_map = {
            MutantStatus.KILLED: "killed", MutantStatus.SURVIVED: "survived",
            MutantStatus.EQUIVALENT: "equivalent", MutantStatus.ERROR: "errors",
            MutantStatus.TIMEOUT: "timeouts",
        }
        for mutant in mutants:
            status, _ = self.run_single(mutant, original_result)
            report.details.append((mutant, status))
            attr = status_map.get(status)
            if attr:
                setattr(report, attr, getattr(report, attr) + 1)
        report.operator_scores = self._compute_operator_scores(report)
        logger.info("Mutation campaign finished.\n%s", report.summary())
        return report

    def higher_order_mutants(self, mutants: List[Mutant], order: int = 2,
                             max_count: int = 100) -> List[Mutant]:
        """Compose *order* first-order mutants into higher-order ones."""
        if order < 2 or len(mutants) < order:
            return []
        combos = list(itertools.combinations(range(len(mutants)), order))
        if len(combos) > max_count:
            chosen = self._rng.choice(len(combos), max_count, replace=False)
            combos = [combos[i] for i in sorted(chosen)]
        ho: List[Mutant] = []
        for combo in combos:
            composed = copy.deepcopy(self.model)
            descs, locs = [], []
            for idx in combo:
                composed = self._merge_mutation(composed, mutants[idx])
                descs.append(mutants[idx].mutation_description)
                locs.append(mutants[idx].mutation_location)
            ho.append(Mutant(self.model, composed, "HigherOrder",
                             " + ".join(descs), ", ".join(locs), order=order))
        return ho

    # -- private helpers ---------------------------------------------------

    def _default_verify(self, model: BioModel,
                        spec: STLFormula) -> VerificationResult:
        cfg = copy.deepcopy(self.config)
        cfg.timeout = min(cfg.timeout, self.timeout)
        return VerificationResult(
            status=VerificationStatus.UNKNOWN, property_name="mutation_test",
            message="Default stub — provide a verify_fn for real verification.")

    def _detect_equivalent(self, mutant: Mutant) -> bool:
        """Heuristic: compare simulated trajectories; if max deviation < ε
        the mutant is likely semantically equivalent."""
        threshold = 1e-4
        try:
            _, y_orig = mutant.original_model.simulate((0, 100), 200)
            _, y_mut = mutant.mutated_model.simulate((0, 100), 200)
            return float(np.max(np.abs(y_orig - y_mut))) < threshold
        except Exception:  # noqa: BLE001
            return self._params_nearly_equal(mutant.original_model,
                                             mutant.mutated_model)

    @staticmethod
    def _params_nearly_equal(a: BioModel, b: BioModel,
                             rtol: float = 1e-6) -> bool:
        try:
            av, bv = a.parameters.values, b.parameters.values
            return set(av) == set(bv) and all(
                np.isclose(av[k], bv[k], rtol=rtol) for k in av)
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _merge_mutation(base: BioModel, mutant: Mutant) -> BioModel:
        """Layer changes from *mutant* onto *base* (best-effort)."""
        merged = copy.deepcopy(base)
        orig, mut = mutant.original_model, mutant.mutated_model
        for pname in orig.parameters.names:
            try:
                ov = orig.parameters.get(pname).value
                mv = mut.parameters.get(pname).value
                if not np.isclose(ov, mv):
                    merged.parameters.get(pname).value = mv
            except (KeyError, ValueError):
                pass
        mut_sp = set(mut.species_names)
        merged._species = {k: v for k, v in merged._species.items() if k in mut_sp}
        mut_rx = {r.name for r in mut.reactions}
        merged._reactions = {k: v for k, v in merged._reactions.items() if k in mut_rx}
        return merged

    @staticmethod
    def _compute_operator_scores(report: MutationReport) -> Dict[str, float]:
        counts: Dict[str, Dict[str, int]] = {}
        for m, s in report.details:
            e = counts.setdefault(m.operator_name, {"total": 0, "killed": 0})
            e["total"] += 1
            if s == MutantStatus.KILLED:
                e["killed"] += 1
        return {op: c["killed"] / c["total"] if c["total"] else 0.0
                for op, c in counts.items()}
