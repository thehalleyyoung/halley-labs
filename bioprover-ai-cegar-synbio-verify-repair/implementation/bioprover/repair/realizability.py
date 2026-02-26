"""Biological realizability constraints for parameter repair.

Ensures that synthesized or repaired parameters correspond to physically
realizable biological designs by checking part compatibility, metabolic
load, physical parameter bounds, and manufacturing constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constraint types
# ---------------------------------------------------------------------------

class ConstraintSeverity(Enum):
    """How critical a constraint violation is."""

    ERROR = auto()    # Hard constraint: design is infeasible
    WARNING = auto()  # Soft constraint: design is suboptimal / risky
    INFO = auto()     # Advisory: worth noting but acceptable


class ConstraintCategory(Enum):
    """Category of biological constraint."""

    PART_COMPATIBILITY = auto()
    METABOLIC_LOAD = auto()
    PHYSICAL_BOUNDS = auto()
    MANUFACTURING = auto()
    GENETIC_STABILITY = auto()


@dataclass
class ConstraintViolation:
    """A single constraint violation."""

    name: str
    category: ConstraintCategory
    severity: ConstraintSeverity
    message: str
    parameter_name: Optional[str] = None
    actual_value: Optional[float] = None
    allowed_range: Optional[Tuple[float, float]] = None

    def __str__(self) -> str:
        tag = self.severity.name
        detail = ""
        if self.parameter_name and self.actual_value is not None:
            detail = f" [{self.parameter_name}={self.actual_value:.4g}"
            if self.allowed_range:
                detail += f", allowed={self.allowed_range}"
            detail += "]"
        return f"[{tag}] {self.name}: {self.message}{detail}"


@dataclass
class BiologicalConstraint:
    """Declarative specification of a biological constraint.

    Parameters
    ----------
    name : str
        Human-readable constraint name.
    check_fn : callable
        ``(params: dict) -> (ok: bool, message: str)``
    category : ConstraintCategory
    severity : ConstraintSeverity
    relaxation_priority : int
        Lower numbers are relaxed first when constraints conflict.
    """

    name: str
    check_fn: Callable[[Dict[str, float]], Tuple[bool, str]]
    category: ConstraintCategory
    severity: ConstraintSeverity = ConstraintSeverity.ERROR
    relaxation_priority: int = 50


# ---------------------------------------------------------------------------
# Physical parameter bounds from literature
# ---------------------------------------------------------------------------

# Typical ranges for common synthetic biology parameters.
# Values drawn from BioNumbers, iGEM Registry, and primary literature.
LITERATURE_BOUNDS: Dict[str, Dict[str, Any]] = {
    "promoter_strength": {
        "units": "RPU",
        "range": (0.01, 15.0),
        "description": "Relative Promoter Unit strength",
    },
    "rbs_efficiency": {
        "units": "au",
        "range": (0.001, 10.0),
        "description": "Ribosome binding site translation initiation rate",
    },
    "protein_degradation_rate": {
        "units": "1/min",
        "range": (0.001, 0.1),
        "description": "ssrA/other tag-mediated degradation rate",
    },
    "mrna_degradation_rate": {
        "units": "1/min",
        "range": (0.02, 0.5),
        "description": "mRNA half-life in range ~1–30 min",
    },
    "hill_coefficient": {
        "units": "dimensionless",
        "range": (1.0, 4.5),
        "description": "Cooperative binding Hill coefficient",
    },
    "dissociation_constant": {
        "units": "nM",
        "range": (0.1, 1000.0),
        "description": "Kd of TF-DNA or protein-ligand interaction",
    },
    "maximal_expression_rate": {
        "units": "nM/min",
        "range": (0.01, 100.0),
        "description": "Maximum protein production rate",
    },
    "basal_expression_rate": {
        "units": "nM/min",
        "range": (0.0, 5.0),
        "description": "Leaky / basal expression rate",
    },
    "maturation_time": {
        "units": "min",
        "range": (5.0, 120.0),
        "description": "Fluorescent protein maturation time",
    },
    "binding_affinity": {
        "units": "1/(nM·min)",
        "range": (1e-4, 10.0),
        "description": "Bimolecular association rate",
    },
    "copy_number": {
        "units": "copies/cell",
        "range": (1, 500),
        "description": "Plasmid copy number",
    },
    "growth_rate": {
        "units": "1/min",
        "range": (0.005, 0.03),
        "description": "Bacterial growth rate (doubling 20-140 min)",
    },
}


# ---------------------------------------------------------------------------
# Realizability report
# ---------------------------------------------------------------------------

@dataclass
class RealizabilityReport:
    """Summary of realizability analysis."""

    feasible: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    warnings: List[ConstraintViolation] = field(default_factory=list)
    info: List[ConstraintViolation] = field(default_factory=list)
    relaxation_order: List[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.violations)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def summary(self) -> str:
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        lines = [
            f"Realizability: {status}",
            f"  Errors: {self.error_count}",
            f"  Warnings: {self.warning_count}",
            f"  Info: {len(self.info)}",
        ]
        for v in self.violations:
            lines.append(f"  ERROR: {v}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        if self.relaxation_order:
            lines.append("  Relaxation order: " + " > ".join(self.relaxation_order))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Realizability checker
# ---------------------------------------------------------------------------

class RealizabilityChecker:
    """Check that parameters correspond to biologically realizable designs.

    Collects constraints from several categories and evaluates them
    against a parameter dictionary.  Provides constraint relaxation
    ordering for infeasible cases.
    """

    def __init__(self) -> None:
        self._constraints: List[BiologicalConstraint] = []
        self._param_type_map: Dict[str, str] = {}

    # -- constraint registration --------------------------------------------

    def add_constraint(self, constraint: BiologicalConstraint) -> None:
        self._constraints.append(constraint)

    def register_parameter_type(self, param_name: str, param_type: str) -> None:
        """Associate *param_name* with a literature parameter type
        (key in ``LITERATURE_BOUNDS``)."""
        self._param_type_map[param_name] = param_type

    def add_default_constraints(self) -> None:
        """Register a standard set of biological constraints."""
        self._add_physical_bounds_constraints()
        self._add_metabolic_load_constraints()
        self._add_part_compatibility_constraints()
        self._add_manufacturing_constraints()
        self._add_genetic_stability_constraints()

    # -- physical bounds (from literature) ----------------------------------

    def _add_physical_bounds_constraints(self) -> None:
        for param_name, param_type in self._param_type_map.items():
            if param_type not in LITERATURE_BOUNDS:
                continue
            entry = LITERATURE_BOUNDS[param_type]
            lo, hi = entry["range"]

            def _make_check(pn: str, low: float, high: float, pt: str):
                def check(params: Dict[str, float]) -> Tuple[bool, str]:
                    if pn not in params:
                        return True, ""
                    v = params[pn]
                    if low <= v <= high:
                        return True, ""
                    return False, (
                        f"{pn} ({pt}) = {v:.4g} outside literature range "
                        f"[{low}, {high}]"
                    )
                return check

            self._constraints.append(BiologicalConstraint(
                name=f"bounds_{param_name}",
                check_fn=_make_check(param_name, lo, hi, param_type),
                category=ConstraintCategory.PHYSICAL_BOUNDS,
                severity=ConstraintSeverity.ERROR,
                relaxation_priority=30,
            ))

    # -- metabolic load -----------------------------------------------------

    def _add_metabolic_load_constraints(self) -> None:
        def check_total_protein(params: Dict[str, float]) -> Tuple[bool, str]:
            prod_keys = [k for k in params if "expression" in k.lower() or "production" in k.lower()]
            if not prod_keys:
                return True, ""
            total = sum(abs(params[k]) for k in prod_keys)
            budget = 200.0  # nM/min total protein budget
            if total <= budget:
                return True, ""
            return False, f"Total protein production {total:.2f} exceeds budget {budget}"

        self._constraints.append(BiologicalConstraint(
            name="metabolic_total_protein",
            check_fn=check_total_protein,
            category=ConstraintCategory.METABOLIC_LOAD,
            severity=ConstraintSeverity.WARNING,
            relaxation_priority=60,
        ))

        def check_growth_impact(params: Dict[str, float]) -> Tuple[bool, str]:
            if "growth_rate" not in params:
                return True, ""
            g = params["growth_rate"]
            if g < 0.005:
                return False, f"Growth rate {g:.4g}/min is dangerously low"
            return True, ""

        self._constraints.append(BiologicalConstraint(
            name="metabolic_growth_rate",
            check_fn=check_growth_impact,
            category=ConstraintCategory.METABOLIC_LOAD,
            severity=ConstraintSeverity.ERROR,
            relaxation_priority=80,
        ))

    # -- part compatibility -------------------------------------------------

    def _add_part_compatibility_constraints(self) -> None:
        def check_promoter_rbs(params: Dict[str, float]) -> Tuple[bool, str]:
            proms = [k for k in params if "promoter" in k.lower()]
            rbss = [k for k in params if "rbs" in k.lower()]
            for p in proms:
                for r in rbss:
                    pv = params[p]
                    rv = params[r]
                    # Very strong promoter + very strong RBS => toxicity risk
                    if pv > 10.0 and rv > 5.0:
                        return False, (
                            f"Strong promoter {p}={pv:.2f} with strong RBS "
                            f"{r}={rv:.2f} risks toxicity"
                        )
            return True, ""

        self._constraints.append(BiologicalConstraint(
            name="promoter_rbs_compatibility",
            check_fn=check_promoter_rbs,
            category=ConstraintCategory.PART_COMPATIBILITY,
            severity=ConstraintSeverity.WARNING,
            relaxation_priority=40,
        ))

    # -- manufacturing constraints ------------------------------------------

    def _add_manufacturing_constraints(self) -> None:
        def check_codon_params(params: Dict[str, float]) -> Tuple[bool, str]:
            # Flag if any parameter implies extreme protein engineering
            for k, v in params.items():
                if "hill" in k.lower() and v > 4.0:
                    return False, (
                        f"Hill coefficient {k}={v:.2f} > 4 is rarely "
                        "achievable without extensive engineering"
                    )
            return True, ""

        self._constraints.append(BiologicalConstraint(
            name="manufacturing_cooperativity",
            check_fn=check_codon_params,
            category=ConstraintCategory.MANUFACTURING,
            severity=ConstraintSeverity.WARNING,
            relaxation_priority=20,
        ))

    # -- genetic stability --------------------------------------------------

    def _add_genetic_stability_constraints(self) -> None:
        def check_copy_stability(params: Dict[str, float]) -> Tuple[bool, str]:
            cn_keys = [k for k in params if "copy" in k.lower()]
            for k in cn_keys:
                if params[k] > 300:
                    return False, (
                        f"Copy number {k}={params[k]:.0f} > 300 risks "
                        "genetic instability"
                    )
            return True, ""

        self._constraints.append(BiologicalConstraint(
            name="genetic_copy_number",
            check_fn=check_copy_stability,
            category=ConstraintCategory.GENETIC_STABILITY,
            severity=ConstraintSeverity.WARNING,
            relaxation_priority=50,
        ))

    # -- main check ---------------------------------------------------------

    def check(self, params: Dict[str, float]) -> RealizabilityReport:
        """Evaluate all constraints against *params*.

        Returns a :class:`RealizabilityReport` with violations, warnings,
        and the recommended relaxation ordering.
        """
        violations: List[ConstraintViolation] = []
        warnings: List[ConstraintViolation] = []
        infos: List[ConstraintViolation] = []

        for c in self._constraints:
            ok, msg = c.check_fn(params)
            if ok:
                continue
            cv = ConstraintViolation(
                name=c.name,
                category=c.category,
                severity=c.severity,
                message=msg,
            )
            if c.severity == ConstraintSeverity.ERROR:
                violations.append(cv)
            elif c.severity == ConstraintSeverity.WARNING:
                warnings.append(cv)
            else:
                infos.append(cv)

        feasible = len(violations) == 0
        relaxation = self._compute_relaxation_order(violations + warnings)

        report = RealizabilityReport(
            feasible=feasible,
            violations=violations,
            warnings=warnings,
            info=infos,
            relaxation_order=relaxation,
        )
        logger.info("Realizability check: %s", "PASS" if feasible else "FAIL")
        return report

    def check_vector(
        self,
        param_vector: np.ndarray,
        param_names: List[str],
    ) -> RealizabilityReport:
        """Convenience: check a numpy parameter vector."""
        params = dict(zip(param_names, param_vector.tolist()))
        return self.check(params)

    def is_feasible(self, params: Dict[str, float]) -> bool:
        """Quick boolean feasibility check (no report construction)."""
        for c in self._constraints:
            if c.severity != ConstraintSeverity.ERROR:
                continue
            ok, _ = c.check_fn(params)
            if not ok:
                return False
        return True

    def feasibility_constraint_fn(
        self, param_names: List[str]
    ) -> Callable[[np.ndarray], bool]:
        """Return a ``params_vector -> bool`` function for use in optimizers."""
        def _check(x: np.ndarray) -> bool:
            return self.is_feasible(dict(zip(param_names, x.tolist())))
        return _check

    # -- relaxation ordering ------------------------------------------------

    def _compute_relaxation_order(
        self, violations: List[ConstraintViolation]
    ) -> List[str]:
        """Order constraint names by relaxation priority (lowest first)."""
        constraint_map = {c.name: c for c in self._constraints}
        scored = []
        for v in violations:
            if v.name in constraint_map:
                scored.append((constraint_map[v.name].relaxation_priority, v.name))
            else:
                scored.append((100, v.name))
        scored.sort()
        return [name for _, name in scored]

    def suggest_relaxation(
        self,
        params: Dict[str, float],
        max_relax: int = 3,
    ) -> List[Tuple[str, str]]:
        """Suggest which constraints to relax first, with rationale.

        Returns list of ``(constraint_name, suggestion)`` pairs.
        """
        report = self.check(params)
        suggestions: List[Tuple[str, str]] = []
        for name in report.relaxation_order[:max_relax]:
            # Find the constraint
            for c in self._constraints:
                if c.name == name:
                    _, msg = c.check_fn(params)
                    if c.category == ConstraintCategory.MANUFACTURING:
                        sug = f"Consider protein engineering or alternative parts: {msg}"
                    elif c.category == ConstraintCategory.METABOLIC_LOAD:
                        sug = f"Reduce expression levels or use low-copy plasmid: {msg}"
                    elif c.category == ConstraintCategory.PHYSICAL_BOUNDS:
                        sug = f"Parameter out of known range—verify experimentally: {msg}"
                    elif c.category == ConstraintCategory.PART_COMPATIBILITY:
                        sug = f"Choose compatible parts or add insulation: {msg}"
                    else:
                        sug = f"Review genetic design for stability: {msg}"
                    suggestions.append((name, sug))
                    break
        return suggestions

    # -- bounds helpers -----------------------------------------------------

    def get_physical_bounds(self, param_name: str) -> Optional[Tuple[float, float]]:
        """Return literature bounds for *param_name*, if registered."""
        ptype = self._param_type_map.get(param_name)
        if ptype and ptype in LITERATURE_BOUNDS:
            return LITERATURE_BOUNDS[ptype]["range"]
        return None

    def tighten_bounds(
        self,
        bounds: List[Tuple[float, float]],
        param_names: List[str],
    ) -> List[Tuple[float, float]]:
        """Intersect optimizer bounds with literature bounds."""
        result = []
        for (lo, hi), name in zip(bounds, param_names):
            lit = self.get_physical_bounds(name)
            if lit is not None:
                lo = max(lo, lit[0])
                hi = min(hi, lit[1])
            result.append((lo, hi))
        return result
