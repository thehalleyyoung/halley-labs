"""
Correctness validation for quotient coalgebras.

Performs differential testing, mutation testing, random property
generation, and detailed discrepancy reporting to validate that
the quotient preserves all relevant properties of the original.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Set, Sequence, Tuple,
)


class PropertyKind(Enum):
    """Classification of temporal properties."""
    SAFETY = auto()
    LIVENESS = auto()
    INVARIANT = auto()
    REACHABILITY = auto()
    FAIRNESS = auto()
    CTL = auto()


class Verdict(Enum):
    """Result of checking a single property."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


class DiscrepancyKind(Enum):
    """Type of mismatch detected between original and quotient."""
    FALSE_POSITIVE = "false_positive"  # quotient satisfies but original does not
    FALSE_NEGATIVE = "false_negative"  # original satisfies but quotient does not
    TIMEOUT_MISMATCH = "timeout_mismatch"
    CRASH = "crash"


@dataclass
class PropertySpec:
    """A property to check on both the original and quotient."""
    name: str
    kind: PropertyKind
    formula: str  # textual representation
    expected: Optional[Verdict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.name,
            "formula": self.formula,
            "expected": self.expected.value if self.expected else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PropertySpec":
        return cls(
            name=d["name"],
            kind=PropertyKind[d["kind"]],
            formula=d["formula"],
            expected=Verdict(d["expected"]) if d.get("expected") else None,
        )


@dataclass
class CheckResult:
    """Result of checking one property on one system."""
    property_spec: PropertySpec
    verdict: Verdict
    elapsed_seconds: float = 0.0
    counterexample: Optional[List[str]] = None
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "property": self.property_spec.name,
            "kind": self.property_spec.kind.name,
            "verdict": self.verdict.value,
            "elapsed_seconds": self.elapsed_seconds,
        }
        if self.counterexample:
            d["counterexample"] = self.counterexample
        if self.error_message:
            d["error_message"] = self.error_message
        return d


@dataclass
class Discrepancy:
    """A mismatch between original and quotient check results."""
    property_spec: PropertySpec
    kind: DiscrepancyKind
    original_verdict: Verdict
    quotient_verdict: Verdict
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property": self.property_spec.name,
            "kind": self.kind.value,
            "original_verdict": self.original_verdict.value,
            "quotient_verdict": self.quotient_verdict.value,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Full report of correctness validation."""
    total_properties: int = 0
    matching: int = 0
    discrepancies: List[Discrepancy] = field(default_factory=list)
    original_results: List[CheckResult] = field(default_factory=list)
    quotient_results: List[CheckResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    mutation_tests_run: int = 0
    mutation_tests_detected: int = 0

    @property
    def correctness_score(self) -> float:
        if self.total_properties == 0:
            return 1.0
        return self.matching / self.total_properties

    @property
    def mutation_detection_rate(self) -> float:
        if self.mutation_tests_run == 0:
            return 1.0
        return self.mutation_tests_detected / self.mutation_tests_run

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_properties": self.total_properties,
            "matching": self.matching,
            "correctness_score": self.correctness_score,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "elapsed_seconds": self.elapsed_seconds,
            "mutation_tests_run": self.mutation_tests_run,
            "mutation_tests_detected": self.mutation_tests_detected,
            "mutation_detection_rate": self.mutation_detection_rate,
        }

    def summary_text(self) -> str:
        lines = [
            f"Correctness score: {self.correctness_score:.2%}",
            f"  Properties checked: {self.total_properties}",
            f"  Matching:           {self.matching}",
            f"  Discrepancies:      {len(self.discrepancies)}",
        ]
        if self.mutation_tests_run > 0:
            lines.append(
                f"  Mutation detection: {self.mutation_detection_rate:.2%} "
                f"({self.mutation_tests_detected}/{self.mutation_tests_run})"
            )
        for d in self.discrepancies:
            lines.append(
                f"  !! {d.kind.value}: {d.property_spec.name} "
                f"(orig={d.original_verdict.value}, quot={d.quotient_verdict.value})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Property checker protocol
# ---------------------------------------------------------------------------

class PropertyChecker:
    """Interface for a pluggable model checker.

    Subclass this and implement ``check`` to integrate with a real
    checker (e.g. TLC, nuXmv, or an internal BFS checker).
    """

    def check(
        self,
        system: Any,
        prop: PropertySpec,
        timeout: float = 60.0,
    ) -> CheckResult:
        raise NotImplementedError

    def check_many(
        self,
        system: Any,
        props: Sequence[PropertySpec],
        timeout: float = 60.0,
    ) -> List[CheckResult]:
        return [self.check(system, p, timeout) for p in props]


class ReachabilityChecker(PropertyChecker):
    """Built-in checker that verifies state reachability properties
    on an FCoalgebra-like object with ``reachable_states()`` method.
    """

    def check(self, system: Any, prop: PropertySpec, timeout: float = 60.0) -> CheckResult:
        t0 = time.monotonic()
        try:
            reachable = system.reachable_states()
            target = prop.formula.strip()
            found = any(target in str(s) for s in reachable)
            verdict = Verdict.SATISFIED if found else Verdict.VIOLATED
            elapsed = time.monotonic() - t0
            return CheckResult(
                property_spec=prop, verdict=verdict, elapsed_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            return CheckResult(
                property_spec=prop,
                verdict=Verdict.UNKNOWN,
                elapsed_seconds=elapsed,
                error_message=str(exc),
            )


# ---------------------------------------------------------------------------
# Random property generation
# ---------------------------------------------------------------------------

def _generate_invariant(states: Sequence[str], rng: random.Random) -> PropertySpec:
    target = rng.choice(states)
    return PropertySpec(
        name=f"inv_{target[:16]}",
        kind=PropertyKind.INVARIANT,
        formula=f"[](state != {target})",
    )


def _generate_reachability(states: Sequence[str], rng: random.Random) -> PropertySpec:
    target = rng.choice(states)
    return PropertySpec(
        name=f"reach_{target[:16]}",
        kind=PropertyKind.REACHABILITY,
        formula=target,
    )


def _generate_safety(states: Sequence[str], rng: random.Random) -> PropertySpec:
    bad = rng.choice(states)
    return PropertySpec(
        name=f"safety_no_{bad[:16]}",
        kind=PropertyKind.SAFETY,
        formula=f"[](NOT {bad})",
    )


def _generate_liveness(states: Sequence[str], rng: random.Random) -> PropertySpec:
    target = rng.choice(states)
    return PropertySpec(
        name=f"live_{target[:16]}",
        kind=PropertyKind.LIVENESS,
        formula=f"<>({target})",
    )


def generate_random_properties(
    states: Sequence[str],
    count: int = 20,
    seed: int = 42,
) -> List[PropertySpec]:
    """Generate a mix of random properties over the given state names."""
    rng = random.Random(seed)
    if not states:
        return []
    generators = [
        _generate_invariant,
        _generate_reachability,
        _generate_safety,
        _generate_liveness,
    ]
    props: List[PropertySpec] = []
    for i in range(count):
        gen = rng.choice(generators)
        props.append(gen(list(states), rng))
    return props


# ---------------------------------------------------------------------------
# Mutation testing
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """Description of a mutation applied to the quotient."""
    name: str
    kind: str  # "add_transition", "remove_transition", "merge_states", etc.
    details: Dict[str, Any] = field(default_factory=dict)


class MutationOperator:
    """Base class for mutation operators on an LTS-like structure."""

    def apply(self, transitions: Dict[str, Dict[str, Set[str]]],
              states: Set[str], rng: random.Random) -> Tuple[Dict[str, Dict[str, Set[str]]], Mutation]:
        raise NotImplementedError


class AddSpuriousTransition(MutationOperator):
    """Add a transition that doesn't exist in the original."""

    def apply(self, transitions, states, rng):
        states_list = list(states)
        if len(states_list) < 2:
            return transitions, Mutation("noop", "noop")
        src = rng.choice(states_list)
        dst = rng.choice(states_list)
        action = f"mut_{rng.randint(0, 999)}"
        new_trans = {s: dict(d) for s, d in transitions.items()}
        new_trans.setdefault(src, {})[action] = {dst}
        return new_trans, Mutation(
            f"add_{src}_{action}_{dst}", "add_transition",
            {"src": src, "action": action, "dst": dst},
        )


class RemoveTransition(MutationOperator):
    """Remove an existing transition."""

    def apply(self, transitions, states, rng):
        candidates = []
        for src, acts in transitions.items():
            for act, dsts in acts.items():
                for d in dsts:
                    candidates.append((src, act, d))
        if not candidates:
            return transitions, Mutation("noop", "noop")
        src, act, dst = rng.choice(candidates)
        new_trans = {}
        for s, acts in transitions.items():
            new_trans[s] = {}
            for a, ds in acts.items():
                new_ds = set(ds)
                if s == src and a == act:
                    new_ds.discard(dst)
                if new_ds:
                    new_trans[s][a] = new_ds
        return new_trans, Mutation(
            f"remove_{src}_{act}_{dst}", "remove_transition",
            {"src": src, "action": act, "dst": dst},
        )


class MergeStates(MutationOperator):
    """Merge two non-equivalent states (introduces false equivalence)."""

    def apply(self, transitions, states, rng):
        states_list = list(states)
        if len(states_list) < 2:
            return transitions, Mutation("noop", "noop")
        s1, s2 = rng.sample(states_list, 2)
        # Replace s2 with s1 everywhere
        new_trans: Dict[str, Dict[str, Set[str]]] = {}
        for src, acts in transitions.items():
            real_src = s1 if src == s2 else src
            new_trans.setdefault(real_src, {})
            for act, dsts in acts.items():
                mapped = {s1 if d == s2 else d for d in dsts}
                existing = new_trans[real_src].get(act, set())
                new_trans[real_src][act] = existing | mapped
        new_states = states - {s2}
        return new_trans, Mutation(
            f"merge_{s1}_{s2}", "merge_states",
            {"kept": s1, "removed": s2},
        )


DEFAULT_MUTATION_OPERATORS: List[MutationOperator] = [
    AddSpuriousTransition(),
    RemoveTransition(),
    MergeStates(),
]


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

class CorrectnessValidator:
    """Orchestrate correctness validation between original and quotient.

    Parameters
    ----------
    checker : PropertyChecker
        The property checker to use for both systems.
    properties : list of PropertySpec
        The properties to verify on both systems.
    timeout : float
        Timeout per individual property check.
    """

    def __init__(
        self,
        checker: Optional[PropertyChecker] = None,
        properties: Optional[List[PropertySpec]] = None,
        timeout: float = 60.0,
        rng_seed: int = 42,
    ) -> None:
        self._checker = checker or ReachabilityChecker()
        self._properties = list(properties) if properties else []
        self._timeout = timeout
        self._rng = random.Random(rng_seed)

    def add_property(self, prop: PropertySpec) -> None:
        self._properties.append(prop)

    def add_properties(self, props: Sequence[PropertySpec]) -> None:
        self._properties.extend(props)

    # -- differential testing ------------------------------------------------

    def _compare_verdicts(
        self,
        orig: CheckResult,
        quot: CheckResult,
    ) -> Optional[Discrepancy]:
        """Return a Discrepancy if the verdicts disagree, else None."""
        ov = orig.verdict
        qv = quot.verdict
        if ov == qv:
            return None
        # Classify
        if ov == Verdict.TIMEOUT or qv == Verdict.TIMEOUT:
            kind = DiscrepancyKind.TIMEOUT_MISMATCH
        elif ov == Verdict.UNKNOWN or qv == Verdict.UNKNOWN:
            kind = DiscrepancyKind.CRASH
        elif ov == Verdict.VIOLATED and qv == Verdict.SATISFIED:
            kind = DiscrepancyKind.FALSE_POSITIVE
        elif ov == Verdict.SATISFIED and qv == Verdict.VIOLATED:
            kind = DiscrepancyKind.FALSE_NEGATIVE
        else:
            kind = DiscrepancyKind.FALSE_NEGATIVE
        details = (
            f"Original: {ov.value}, Quotient: {qv.value}. "
            f"Property: {orig.property_spec.formula}"
        )
        return Discrepancy(
            property_spec=orig.property_spec,
            kind=kind,
            original_verdict=ov,
            quotient_verdict=qv,
            details=details,
        )

    def validate(
        self,
        original: Any,
        quotient: Any,
    ) -> ValidationReport:
        """Run all properties on both systems and compare."""
        t0 = time.monotonic()
        report = ValidationReport()
        report.total_properties = len(self._properties)

        orig_results = self._checker.check_many(
            original, self._properties, self._timeout
        )
        quot_results = self._checker.check_many(
            quotient, self._properties, self._timeout
        )
        report.original_results = orig_results
        report.quotient_results = quot_results

        for o, q in zip(orig_results, quot_results):
            disc = self._compare_verdicts(o, q)
            if disc is None:
                report.matching += 1
            else:
                report.discrepancies.append(disc)

        report.elapsed_seconds = time.monotonic() - t0
        return report

    # -- fuzz testing --------------------------------------------------------

    def fuzz_validate(
        self,
        original: Any,
        quotient: Any,
        state_names: Sequence[str],
        fuzz_count: int = 50,
    ) -> ValidationReport:
        """Generate random properties and validate."""
        random_props = generate_random_properties(
            list(state_names), count=fuzz_count, seed=self._rng.randint(0, 2**31),
        )
        self._properties.extend(random_props)
        return self.validate(original, quotient)

    # -- mutation testing ----------------------------------------------------

    def mutation_test(
        self,
        original: Any,
        quotient_transitions: Dict[str, Dict[str, Set[str]]],
        quotient_states: Set[str],
        build_quotient: Callable,
        mutation_count: int = 10,
        operators: Optional[List[MutationOperator]] = None,
    ) -> ValidationReport:
        """Apply mutations to the quotient and verify that the checker
        detects the introduced bugs.

        ``build_quotient`` should be a callable that takes
        (transitions, states) and returns a system object suitable
        for the property checker.
        """
        ops = operators or DEFAULT_MUTATION_OPERATORS
        report = ValidationReport()
        report.total_properties = len(self._properties)

        # First, get original results
        orig_results = self._checker.check_many(
            original, self._properties, self._timeout
        )
        report.original_results = orig_results

        detected = 0
        total = 0
        for _ in range(mutation_count):
            op = self._rng.choice(ops)
            mut_trans, mutation = op.apply(
                quotient_transitions, quotient_states, self._rng,
            )
            if mutation.kind == "noop":
                continue
            total += 1
            try:
                mutant = build_quotient(mut_trans, quotient_states)
                mut_results = self._checker.check_many(
                    mutant, self._properties, self._timeout
                )
                # Check if any discrepancy was found
                found_disc = False
                for o, m in zip(orig_results, mut_results):
                    if o.verdict != m.verdict:
                        found_disc = True
                        break
                if found_disc:
                    detected += 1
            except Exception:
                detected += 1  # crash counts as detection

        report.mutation_tests_run = total
        report.mutation_tests_detected = detected

        # Also do the normal comparison for the unmodified quotient
        quot_results = self._checker.check_many(
            build_quotient(quotient_transitions, quotient_states),
            self._properties,
            self._timeout,
        )
        report.quotient_results = quot_results
        for o, q in zip(orig_results, quot_results):
            disc = self._compare_verdicts(o, q)
            if disc is None:
                report.matching += 1
            else:
                report.discrepancies.append(disc)

        return report

    # -- report generation ---------------------------------------------------

    def detailed_report(
        self,
        report: ValidationReport,
    ) -> str:
        """Generate a detailed human-readable discrepancy report."""
        lines = [report.summary_text(), ""]
        if report.discrepancies:
            lines.append("=== Detailed Discrepancies ===")
            for i, d in enumerate(report.discrepancies, 1):
                lines.append(f"\n--- Discrepancy {i} ---")
                lines.append(f"  Property: {d.property_spec.name}")
                lines.append(f"  Kind:     {d.property_spec.kind.name}")
                lines.append(f"  Formula:  {d.property_spec.formula}")
                lines.append(f"  Type:     {d.kind.value}")
                lines.append(f"  Original: {d.original_verdict.value}")
                lines.append(f"  Quotient: {d.quotient_verdict.value}")
                if d.details:
                    lines.append(f"  Details:  {d.details}")
        else:
            lines.append("No discrepancies found.")
        return "\n".join(lines)
