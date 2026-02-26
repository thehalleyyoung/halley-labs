"""
Differential testing for CoaCert-TLA property checking.

Compares model-checking results on the original coalgebraic system
versus its quotient (bisimulation collapse).  Any discrepancy indicates
either a bug in the quotienting procedure or a property that is not
preserved by the morphism.

Includes random property generation for fuzz testing and statistical
summary reporting.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

from .ctl_star import CTLStarChecker, CTLCheckResult, KripkeAdapter
from .liveness import (
    FairnessSpec,
    LivenessChecker,
    LivenessCheckResult,
    LivenessKind,
    LivenessProperty,
    make_eventually_always,
    make_infinitely_often,
    make_leads_to,
)
from .safety import (
    SafetyCheckResult,
    SafetyChecker,
    SafetyKind,
    SafetyProperty,
    make_ap_invariant,
    make_exclusion_invariant,
)
from .temporal_logic import (
    AG,
    AF,
    EF,
    EG,
    EU,
    AX,
    And,
    Atomic,
    ExistsPath,
    FalseFormula,
    Finally,
    ForallPath,
    Globally,
    Implies,
    Next,
    Not,
    Or,
    TemporalFormula,
    TrueFormula,
    Until,
    is_ctl,
    is_ltl,
    is_stuttering_invariant,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Discrepancy reporting
# ============================================================================

class DiscrepancyKind(Enum):
    """Type of disagreement between original and quotient."""
    SAFETY_MISMATCH = auto()
    LIVENESS_MISMATCH = auto()
    CTL_MISMATCH = auto()
    SATISFYING_SET_DIFF = auto()


@dataclass
class Discrepancy:
    """A single discrepancy between original and quotient checking."""
    kind: DiscrepancyKind
    property_name: str
    original_holds: bool
    quotient_holds: bool
    details: str = ""
    formula: Optional[TemporalFormula] = None

    def summary(self) -> str:
        orig = "PASS" if self.original_holds else "FAIL"
        quot = "PASS" if self.quotient_holds else "FAIL"
        msg = f"DISCREPANCY [{self.kind.name}] {self.property_name}: "
        msg += f"original={orig}, quotient={quot}"
        if self.details:
            msg += f"\n  {self.details}"
        return msg


# ============================================================================
# Statistical summary
# ============================================================================

@dataclass
class DifferentialStats:
    """Statistical summary of differential testing results."""
    total_tests: int = 0
    agreements: int = 0
    discrepancies: int = 0
    safety_tests: int = 0
    safety_agreements: int = 0
    liveness_tests: int = 0
    liveness_agreements: int = 0
    ctl_tests: int = 0
    ctl_agreements: int = 0
    elapsed_seconds: float = 0.0
    discrepancy_list: List[Discrepancy] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        if self.total_tests == 0:
            return 1.0
        return self.agreements / self.total_tests

    @property
    def confidence(self) -> float:
        """Bayesian confidence that the quotient is correct.

        Uses a simple Beta(agreements+1, discrepancies+1) model.
        Returns the mean of the posterior.
        """
        alpha = self.agreements + 1
        beta = self.discrepancies + 1
        return alpha / (alpha + beta)

    def summary(self) -> str:
        lines = [
            f"Differential Testing Summary",
            f"  Total tests:     {self.total_tests}",
            f"  Agreements:      {self.agreements}",
            f"  Discrepancies:   {self.discrepancies}",
            f"  Agreement rate:  {self.agreement_rate:.2%}",
            f"  Confidence:      {self.confidence:.4f}",
            f"  Time elapsed:    {self.elapsed_seconds:.2f}s",
            f"  Safety:          {self.safety_agreements}/{self.safety_tests}",
            f"  Liveness:        {self.liveness_agreements}/{self.liveness_tests}",
            f"  CTL:             {self.ctl_agreements}/{self.ctl_tests}",
        ]
        if self.discrepancy_list:
            lines.append("  Discrepancies:")
            for d in self.discrepancy_list:
                lines.append(f"    - {d.summary()}")
        return "\n".join(lines)


# ============================================================================
# Random property generators
# ============================================================================

class RandomPropertyGenerator:
    """Generate random temporal logic formulas for fuzz testing."""

    def __init__(
        self,
        atomic_props: List[str],
        seed: Optional[int] = None,
    ) -> None:
        self._aps = atomic_props
        self._rng = random.Random(seed)

    def random_state_predicate(self) -> TemporalFormula:
        """Generate a random Boolean combination of atomic propositions."""
        depth = self._rng.randint(0, 3)
        return self._random_bool_formula(depth)

    def _random_bool_formula(self, depth: int) -> TemporalFormula:
        if depth <= 0 or self._rng.random() < 0.3:
            return self._random_atomic()
        op = self._rng.choice(["and", "or", "not", "implies"])
        if op == "not":
            return Not(self._random_bool_formula(depth - 1))
        if op == "and":
            return And(
                self._random_bool_formula(depth - 1),
                self._random_bool_formula(depth - 1),
            )
        if op == "or":
            return Or(
                self._random_bool_formula(depth - 1),
                self._random_bool_formula(depth - 1),
            )
        return Implies(
            self._random_bool_formula(depth - 1),
            self._random_bool_formula(depth - 1),
        )

    def _random_atomic(self) -> TemporalFormula:
        coin = self._rng.random()
        if coin < 0.1:
            return TrueFormula()
        if coin < 0.15:
            return FalseFormula()
        return Atomic(self._rng.choice(self._aps))

    def random_ctl_formula(self, max_depth: int = 3) -> TemporalFormula:
        """Generate a random CTL formula."""
        return self._random_ctl(max_depth)

    def _random_ctl(self, depth: int) -> TemporalFormula:
        if depth <= 0:
            return self._random_atomic()

        kind = self._rng.choice([
            "ap", "not", "and", "or",
            "ex", "ef", "eg", "eu",
            "ax", "af", "ag", "au",
        ])

        if kind == "ap":
            return self._random_atomic()
        if kind == "not":
            return Not(self._random_ctl(depth - 1))
        if kind == "and":
            return And(self._random_ctl(depth - 1), self._random_ctl(depth - 1))
        if kind == "or":
            return Or(self._random_ctl(depth - 1), self._random_ctl(depth - 1))
        if kind == "ex":
            return ExistsPath(Next(self._random_ctl(depth - 1)))
        if kind == "ef":
            return ExistsPath(Finally(self._random_ctl(depth - 1)))
        if kind == "eg":
            return ExistsPath(Globally(self._random_ctl(depth - 1)))
        if kind == "eu":
            return ExistsPath(Until(
                self._random_ctl(depth - 1), self._random_ctl(depth - 1)
            ))
        if kind == "ax":
            return ForallPath(Next(self._random_ctl(depth - 1)))
        if kind == "af":
            return ForallPath(Finally(self._random_ctl(depth - 1)))
        if kind == "ag":
            return ForallPath(Globally(self._random_ctl(depth - 1)))
        # au
        return ForallPath(Until(
            self._random_ctl(depth - 1), self._random_ctl(depth - 1)
        ))

    def random_ltl_formula(self, max_depth: int = 3) -> TemporalFormula:
        """Generate a random LTL formula (no path quantifiers)."""
        return self._random_ltl(max_depth)

    def _random_ltl(self, depth: int) -> TemporalFormula:
        if depth <= 0:
            return self._random_atomic()

        kind = self._rng.choice([
            "ap", "not", "and", "or",
            "next", "finally", "globally", "until",
        ])
        if kind == "ap":
            return self._random_atomic()
        if kind == "not":
            return Not(self._random_ltl(depth - 1))
        if kind == "and":
            return And(self._random_ltl(depth - 1), self._random_ltl(depth - 1))
        if kind == "or":
            return Or(self._random_ltl(depth - 1), self._random_ltl(depth - 1))
        if kind == "next":
            return Next(self._random_ltl(depth - 1))
        if kind == "finally":
            return Finally(self._random_ltl(depth - 1))
        if kind == "globally":
            return Globally(self._random_ltl(depth - 1))
        # until
        return Until(self._random_ltl(depth - 1), self._random_ltl(depth - 1))

    def random_safety_property(self) -> SafetyProperty:
        """Generate a random state-invariant safety property."""
        pred_formula = self.random_state_predicate()
        prop_name = f"Inv({pred_formula})"

        def evaluate(state: str, labels: FrozenSet[str], f=pred_formula) -> bool:
            return _eval_formula_on_labels(f, labels)

        return SafetyProperty(
            name=prop_name,
            kind=SafetyKind.STATE_INVARIANT,
            predicate=evaluate,
            description=f"Random invariant: {pred_formula}",
        )

    def random_liveness_property(self) -> LivenessProperty:
        """Generate a random liveness property."""
        kind = self._rng.choice([
            LivenessKind.INFINITELY_OFTEN,
            LivenessKind.EVENTUALLY_ALWAYS,
            LivenessKind.LEADS_TO,
        ])
        if kind == LivenessKind.LEADS_TO:
            phi = self._random_atomic()
            psi = self._random_atomic()
            return LivenessProperty(
                name=f"{phi} ⤳ {psi}",
                kind=kind,
                phi=phi,
                psi=psi,
            )
        else:
            phi = self._random_atomic()
            prefix = "□◇" if kind == LivenessKind.INFINITELY_OFTEN else "◇□"
            return LivenessProperty(
                name=f"{prefix}{phi}",
                kind=kind,
                phi=phi,
            )


def _eval_formula_on_labels(f: TemporalFormula, labels: FrozenSet[str]) -> bool:
    """Evaluate a propositional formula on a set of labels."""
    if isinstance(f, TrueFormula):
        return True
    if isinstance(f, FalseFormula):
        return False
    if isinstance(f, Atomic):
        return f.name in labels
    if isinstance(f, Not):
        return not _eval_formula_on_labels(f.child, labels)
    if isinstance(f, And):
        return (_eval_formula_on_labels(f.left, labels) and
                _eval_formula_on_labels(f.right, labels))
    if isinstance(f, Or):
        return (_eval_formula_on_labels(f.left, labels) or
                _eval_formula_on_labels(f.right, labels))
    if isinstance(f, Implies):
        return (not _eval_formula_on_labels(f.left, labels) or
                _eval_formula_on_labels(f.right, labels))
    return True


# ============================================================================
# Benchmark property suite
# ============================================================================

class BenchmarkPropertySuite:
    """Systematic property generation for known spec patterns."""

    def __init__(self, kripke: KripkeAdapter) -> None:
        self._kripke = kripke
        self._aps = self._collect_aps()

    def _collect_aps(self) -> List[str]:
        """Collect all atomic propositions from the system."""
        aps: Set[str] = set()
        for labels in self._kripke.labels.values():
            aps.update(labels)
        return sorted(aps)

    def safety_properties(self) -> List[SafetyProperty]:
        """Generate systematic safety properties."""
        props: List[SafetyProperty] = []

        # Single AP invariants
        for ap in self._aps:
            props.append(make_ap_invariant(ap))

        # Mutual exclusion for all pairs
        for i, a in enumerate(self._aps):
            for b in self._aps[i + 1:]:
                props.append(make_exclusion_invariant(a, b))

        # Disjunction invariants: at least one AP holds
        if len(self._aps) >= 2:
            def at_least_one(state: str, labels: FrozenSet[str]) -> bool:
                return bool(labels)

            props.append(SafetyProperty(
                name="AtLeastOneAP",
                kind=SafetyKind.STATE_INVARIANT,
                predicate=at_least_one,
                description="At least one atomic proposition holds",
            ))

        return props

    def liveness_properties(self) -> List[LivenessProperty]:
        """Generate systematic liveness properties."""
        props: List[LivenessProperty] = []

        for ap in self._aps:
            props.append(make_infinitely_often(f"□◇{ap}", ap))
            props.append(make_eventually_always(f"◇□{ap}", ap))

        for i, a in enumerate(self._aps):
            for b in self._aps[i + 1:]:
                props.append(make_leads_to(f"{a}⤳{b}", a, b))

        return props

    def ctl_properties(self) -> List[TemporalFormula]:
        """Generate systematic CTL formulas."""
        formulas: List[TemporalFormula] = []

        for ap in self._aps:
            a = Atomic(ap)
            formulas.append(AG(a))
            formulas.append(AF(a))
            formulas.append(EF(a))
            formulas.append(EG(a))

        for i, ap1 in enumerate(self._aps):
            for ap2 in self._aps[i + 1:]:
                a = Atomic(ap1)
                b = Atomic(ap2)
                formulas.append(AG(Implies(a, AF(b))))
                formulas.append(EU(a, b))

        return formulas


# ============================================================================
# Differential tester
# ============================================================================

class DifferentialTester:
    """Compare property checking results on original vs quotient systems.

    Runs safety, liveness, and CTL checks on both systems and reports
    any discrepancies.
    """

    def __init__(
        self,
        original_coalg: object,
        quotient_coalg: object,
        projection: Mapping[str, str],
        fairness: Optional[List[FairnessSpec]] = None,
    ) -> None:
        self._orig = original_coalg
        self._quot = quotient_coalg
        self._proj = projection
        self._fairness = fairness or []

        self._orig_safety = SafetyChecker(original_coalg)
        self._quot_safety = SafetyChecker(quotient_coalg)
        self._orig_ctl = CTLStarChecker(original_coalg)
        self._quot_ctl = CTLStarChecker(quotient_coalg)
        self._orig_live = LivenessChecker(original_coalg, fairness)
        self._quot_live = LivenessChecker(quotient_coalg, fairness)

    # -- Full differential suite -------------------------------------------

    def run_full_suite(
        self,
        safety_props: Optional[List[SafetyProperty]] = None,
        liveness_props: Optional[List[LivenessProperty]] = None,
        ctl_formulas: Optional[List[TemporalFormula]] = None,
    ) -> DifferentialStats:
        """Run differential testing with provided properties."""
        start = time.monotonic()
        stats = DifferentialStats()

        if safety_props:
            self._test_safety(safety_props, stats)
        if liveness_props:
            self._test_liveness(liveness_props, stats)
        if ctl_formulas:
            self._test_ctl(ctl_formulas, stats)

        stats.elapsed_seconds = time.monotonic() - start
        logger.info("Differential testing complete: %s", stats.summary())
        return stats

    def run_benchmark_suite(self) -> DifferentialStats:
        """Run differential testing with auto-generated benchmark properties."""
        suite = BenchmarkPropertySuite(self._orig_ctl.kripke)
        return self.run_full_suite(
            safety_props=suite.safety_properties(),
            liveness_props=suite.liveness_properties(),
            ctl_formulas=suite.ctl_properties(),
        )

    def run_fuzz_suite(
        self,
        n_safety: int = 20,
        n_liveness: int = 20,
        n_ctl: int = 20,
        seed: Optional[int] = None,
    ) -> DifferentialStats:
        """Run differential testing with randomly generated properties."""
        aps = sorted(set().union(
            *(labels for labels in self._orig_ctl.kripke.labels.values())
        ))
        if not aps:
            aps = ["p", "q", "r"]

        gen = RandomPropertyGenerator(aps, seed=seed)
        safety_props = [gen.random_safety_property() for _ in range(n_safety)]
        liveness_props = [gen.random_liveness_property() for _ in range(n_liveness)]
        ctl_formulas = [gen.random_ctl_formula() for _ in range(n_ctl)]

        return self.run_full_suite(
            safety_props=safety_props,
            liveness_props=liveness_props,
            ctl_formulas=ctl_formulas,
        )

    # -- Individual test categories -----------------------------------------

    def _test_safety(
        self,
        props: List[SafetyProperty],
        stats: DifferentialStats,
    ) -> None:
        """Differentially test safety properties."""
        for prop in props:
            stats.total_tests += 1
            stats.safety_tests += 1

            orig_result = self._orig_safety.check_invariant(prop)
            quot_result = self._quot_safety.check_invariant(prop)

            if orig_result.holds == quot_result.holds:
                stats.agreements += 1
                stats.safety_agreements += 1
            else:
                stats.discrepancies += 1
                disc = Discrepancy(
                    kind=DiscrepancyKind.SAFETY_MISMATCH,
                    property_name=prop.name,
                    original_holds=orig_result.holds,
                    quotient_holds=quot_result.holds,
                    details=self._safety_detail(orig_result, quot_result),
                )
                stats.discrepancy_list.append(disc)
                logger.warning(disc.summary())

    def _test_liveness(
        self,
        props: List[LivenessProperty],
        stats: DifferentialStats,
    ) -> None:
        """Differentially test liveness properties."""
        for prop in props:
            stats.total_tests += 1
            stats.liveness_tests += 1

            orig_result = self._orig_live.check(prop)
            quot_result = self._quot_live.check(prop)

            if orig_result.holds == quot_result.holds:
                stats.agreements += 1
                stats.liveness_agreements += 1
            else:
                stats.discrepancies += 1
                disc = Discrepancy(
                    kind=DiscrepancyKind.LIVENESS_MISMATCH,
                    property_name=prop.name,
                    original_holds=orig_result.holds,
                    quotient_holds=quot_result.holds,
                    details=self._liveness_detail(orig_result, quot_result),
                )
                stats.discrepancy_list.append(disc)
                logger.warning(disc.summary())

    def _test_ctl(
        self,
        formulas: List[TemporalFormula],
        stats: DifferentialStats,
    ) -> None:
        """Differentially test CTL/CTL* formulas."""
        for formula in formulas:
            stats.total_tests += 1
            stats.ctl_tests += 1

            orig_result = self._orig_ctl.check(formula)
            quot_result = self._quot_ctl.check(formula)

            if orig_result.holds == quot_result.holds:
                stats.agreements += 1
                stats.ctl_agreements += 1
            else:
                stats.discrepancies += 1
                # Compute satisfying set difference
                diff_detail = self._ctl_detail(orig_result, quot_result)
                disc = Discrepancy(
                    kind=DiscrepancyKind.CTL_MISMATCH,
                    property_name=str(formula),
                    original_holds=orig_result.holds,
                    quotient_holds=quot_result.holds,
                    details=diff_detail,
                    formula=formula,
                )
                stats.discrepancy_list.append(disc)
                logger.warning(disc.summary())

    # -- Detail generation -------------------------------------------------

    @staticmethod
    def _safety_detail(orig: SafetyCheckResult, quot: SafetyCheckResult) -> str:
        parts: List[str] = []
        if orig.violating_state:
            parts.append(f"orig violating state: {orig.violating_state}")
        if quot.violating_state:
            parts.append(f"quot violating state: {quot.violating_state}")
        parts.append(f"orig checked {orig.states_checked} states")
        parts.append(f"quot checked {quot.states_checked} states")
        return "; ".join(parts)

    @staticmethod
    def _liveness_detail(orig: LivenessCheckResult, quot: LivenessCheckResult) -> str:
        parts: List[str] = []
        parts.append(f"orig method={orig.method}, quot method={quot.method}")
        parts.append(f"orig SCCs: {orig.accepting_sccs_count}/{orig.total_sccs_count}")
        parts.append(f"quot SCCs: {quot.accepting_sccs_count}/{quot.total_sccs_count}")
        return "; ".join(parts)

    @staticmethod
    def _ctl_detail(orig: CTLCheckResult, quot: CTLCheckResult) -> str:
        o_sat = orig.satisfying_states
        q_sat = quot.satisfying_states
        only_orig = o_sat - q_sat
        only_quot = q_sat - o_sat
        parts: List[str] = []
        parts.append(f"|orig_sat|={len(o_sat)}, |quot_sat|={len(q_sat)}")
        if only_orig:
            parts.append(f"only in original: {sorted(only_orig)[:5]}")
        if only_quot:
            parts.append(f"only in quotient: {sorted(only_quot)[:5]}")
        return "; ".join(parts)

    # -- Single property comparison -----------------------------------------

    def compare_safety(
        self,
        prop: SafetyProperty,
    ) -> Tuple[SafetyCheckResult, SafetyCheckResult, bool]:
        """Compare a single safety property on original vs quotient."""
        orig = self._orig_safety.check_invariant(prop)
        quot = self._quot_safety.check_invariant(prop)
        return orig, quot, orig.holds == quot.holds

    def compare_liveness(
        self,
        prop: LivenessProperty,
    ) -> Tuple[LivenessCheckResult, LivenessCheckResult, bool]:
        """Compare a single liveness property on original vs quotient."""
        orig = self._orig_live.check(prop)
        quot = self._quot_live.check(prop)
        return orig, quot, orig.holds == quot.holds

    def compare_ctl(
        self,
        formula: TemporalFormula,
    ) -> Tuple[CTLCheckResult, CTLCheckResult, bool]:
        """Compare a single CTL formula on original vs quotient."""
        orig = self._orig_ctl.check(formula)
        quot = self._quot_ctl.check(formula)
        return orig, quot, orig.holds == quot.holds

    # -- Confidence computation ---------------------------------------------

    @staticmethod
    def compute_confidence(stats: DifferentialStats, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> Dict[str, float]:
        """Compute Bayesian confidence levels for various property classes.

        Uses a Beta-Binomial model:
          posterior_alpha = prior_alpha + agreements
          posterior_beta  = prior_beta  + discrepancies

        Returns a dict of confidence values.
        """
        result: Dict[str, float] = {}

        def beta_mean(a: int, d: int) -> float:
            alpha = prior_alpha + a
            beta = prior_beta + d
            return alpha / (alpha + beta)

        result["overall"] = beta_mean(stats.agreements, stats.discrepancies)
        result["safety"] = beta_mean(
            stats.safety_agreements,
            stats.safety_tests - stats.safety_agreements,
        )
        result["liveness"] = beta_mean(
            stats.liveness_agreements,
            stats.liveness_tests - stats.liveness_agreements,
        )
        result["ctl"] = beta_mean(
            stats.ctl_agreements,
            stats.ctl_tests - stats.ctl_agreements,
        )
        return result
