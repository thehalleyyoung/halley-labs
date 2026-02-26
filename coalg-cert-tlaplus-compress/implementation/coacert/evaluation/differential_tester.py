"""
Differential testing engine for CoaCert.

Compares the explicit exploration of an original system with the
compressed quotient on a state-by-state, transition-by-transition,
and property-by-property basis.  Includes random-walk comparison,
coverage analysis, and bug classification.
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

from .correctness import (
    CheckResult,
    Discrepancy,
    DiscrepancyKind,
    PropertyChecker,
    PropertySpec,
    Verdict,
)


class BugKind(Enum):
    """Classification of bugs found via differential testing."""
    FALSE_POSITIVE = auto()
    FALSE_NEGATIVE = auto()
    HASH_COLLISION = auto()
    MISSING_STATE = auto()
    EXTRA_STATE = auto()
    MISSING_TRANSITION = auto()
    EXTRA_TRANSITION = auto()
    LABEL_MISMATCH = auto()


@dataclass
class StateDiff:
    """Difference in a single state between original and quotient."""
    state: str
    in_original: bool
    in_quotient: bool
    original_labels: FrozenSet[str] = field(default_factory=frozenset)
    quotient_labels: FrozenSet[str] = field(default_factory=frozenset)
    bug_kind: Optional[BugKind] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "state": self.state,
            "in_original": self.in_original,
            "in_quotient": self.in_quotient,
        }
        if self.original_labels:
            d["original_labels"] = sorted(self.original_labels)
        if self.quotient_labels:
            d["quotient_labels"] = sorted(self.quotient_labels)
        if self.bug_kind:
            d["bug_kind"] = self.bug_kind.name
        return d


@dataclass
class TransitionDiff:
    """Difference in a single transition."""
    src: str
    action: str
    dst: str
    in_original: bool
    in_quotient: bool
    bug_kind: Optional[BugKind] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "src": self.src,
            "action": self.action,
            "dst": self.dst,
            "in_original": self.in_original,
            "in_quotient": self.in_quotient,
        }
        if self.bug_kind:
            d["bug_kind"] = self.bug_kind.name
        return d


@dataclass
class RandomWalkResult:
    """Outcome of a random walk comparison."""
    walk_length: int
    original_trace: List[str]
    quotient_trace: List[str]
    divergence_step: int = -1  # -1 means no divergence
    divergence_detail: str = ""

    @property
    def diverged(self) -> bool:
        return self.divergence_step >= 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "walk_length": self.walk_length,
            "diverged": self.diverged,
            "divergence_step": self.divergence_step,
            "divergence_detail": self.divergence_detail,
            "original_trace_len": len(self.original_trace),
            "quotient_trace_len": len(self.quotient_trace),
        }


@dataclass
class CoverageInfo:
    """Coverage statistics from differential testing."""
    total_original_states: int = 0
    tested_original_states: int = 0
    total_original_transitions: int = 0
    tested_original_transitions: int = 0
    total_quotient_states: int = 0
    tested_quotient_states: int = 0
    total_quotient_transitions: int = 0
    tested_quotient_transitions: int = 0

    @property
    def original_state_coverage(self) -> float:
        if self.total_original_states == 0:
            return 1.0
        return self.tested_original_states / self.total_original_states

    @property
    def original_transition_coverage(self) -> float:
        if self.total_original_transitions == 0:
            return 1.0
        return self.tested_original_transitions / self.total_original_transitions

    @property
    def quotient_state_coverage(self) -> float:
        if self.total_quotient_states == 0:
            return 1.0
        return self.tested_quotient_states / self.total_quotient_states

    @property
    def quotient_transition_coverage(self) -> float:
        if self.total_quotient_transitions == 0:
            return 1.0
        return self.tested_quotient_transitions / self.total_quotient_transitions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_state_coverage": self.original_state_coverage,
            "original_transition_coverage": self.original_transition_coverage,
            "quotient_state_coverage": self.quotient_state_coverage,
            "quotient_transition_coverage": self.quotient_transition_coverage,
            "total_original_states": self.total_original_states,
            "tested_original_states": self.tested_original_states,
            "total_original_transitions": self.total_original_transitions,
            "tested_original_transitions": self.tested_original_transitions,
        }


@dataclass
class DifferentialTestReport:
    """Complete report from a differential testing session."""
    state_diffs: List[StateDiff] = field(default_factory=list)
    transition_diffs: List[TransitionDiff] = field(default_factory=list)
    property_discrepancies: List[Discrepancy] = field(default_factory=list)
    random_walk_results: List[RandomWalkResult] = field(default_factory=list)
    coverage: CoverageInfo = field(default_factory=CoverageInfo)
    bugs_found: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def total_diffs(self) -> int:
        return len(self.state_diffs) + len(self.transition_diffs)

    @property
    def walks_diverged(self) -> int:
        return sum(1 for w in self.random_walk_results if w.diverged)

    @property
    def pass_rate(self) -> float:
        if not self.random_walk_results:
            return 1.0
        return 1.0 - (self.walks_diverged / len(self.random_walk_results))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_diffs": [d.to_dict() for d in self.state_diffs],
            "transition_diffs": [d.to_dict() for d in self.transition_diffs],
            "property_discrepancies": [d.to_dict() for d in self.property_discrepancies],
            "random_walks": {
                "total": len(self.random_walk_results),
                "diverged": self.walks_diverged,
                "pass_rate": self.pass_rate,
            },
            "coverage": self.coverage.to_dict(),
            "bugs_found": self.bugs_found,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def summary_text(self) -> str:
        lines = [
            f"Differential Test Report",
            f"  State diffs:       {len(self.state_diffs)}",
            f"  Transition diffs:  {len(self.transition_diffs)}",
            f"  Property mismatches: {len(self.property_discrepancies)}",
            f"  Random walks:      {len(self.random_walk_results)} "
            f"({self.walks_diverged} diverged, pass={self.pass_rate:.2%})",
            f"  Bugs classified:   {len(self.bugs_found)}",
            f"  Coverage (orig):   states={self.coverage.original_state_coverage:.1%} "
            f"trans={self.coverage.original_transition_coverage:.1%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LTS representation for comparison (lightweight)
# ---------------------------------------------------------------------------

@dataclass
class LTSSnapshot:
    """Lightweight snapshot of an LTS for comparison purposes."""
    states: Set[str]
    initial: str
    transitions: Dict[str, Dict[str, Set[str]]]  # src -> action -> {dst}
    labels: Dict[str, FrozenSet[str]]  # state -> {atomic props}

    @property
    def transition_count(self) -> int:
        return sum(
            len(dsts) for acts in self.transitions.values() for dsts in acts.values()
        )

    @classmethod
    def from_coalgebra(cls, coalgebra: Any) -> "LTSSnapshot":
        """Build from an FCoalgebra-like object."""
        states: Set[str] = set()
        transitions: Dict[str, Dict[str, Set[str]]] = {}
        labels: Dict[str, FrozenSet[str]] = {}

        if hasattr(coalgebra, "states"):
            for s in coalgebra.states:
                name = str(s) if not isinstance(s, str) else s
                states.add(name)
                if hasattr(s, "propositions"):
                    labels[name] = frozenset(str(p) for p in s.propositions)
                if hasattr(s, "successors"):
                    transitions[name] = {}
                    for act, dsts in s.successors.items():
                        transitions[name][str(act)] = {str(d) for d in dsts}

        initial = ""
        if hasattr(coalgebra, "initial_state"):
            initial = str(coalgebra.initial_state)
        elif states:
            initial = min(states)

        return cls(
            states=states, initial=initial,
            transitions=transitions, labels=labels,
        )

    @classmethod
    def from_explicit(
        cls,
        states: Set[str],
        initial: str,
        transitions: Dict[str, Dict[str, Set[str]]],
        labels: Optional[Dict[str, FrozenSet[str]]] = None,
    ) -> "LTSSnapshot":
        return cls(
            states=states, initial=initial,
            transitions=transitions,
            labels=labels or {},
        )


# ---------------------------------------------------------------------------
# Quotient mapping
# ---------------------------------------------------------------------------

@dataclass
class QuotientMapping:
    """Maps original states to quotient block representatives."""
    state_to_block: Dict[str, str]  # original state -> block representative
    block_to_states: Dict[str, Set[str]]  # block -> {original states}

    def map_state(self, state: str) -> str:
        return self.state_to_block.get(state, state)

    def block_states(self, block: str) -> Set[str]:
        return self.block_to_states.get(block, set())

    @classmethod
    def from_partition(
        cls, partition: Sequence[Set[str]]
    ) -> "QuotientMapping":
        s2b: Dict[str, str] = {}
        b2s: Dict[str, Set[str]] = {}
        for block in partition:
            rep = min(block)
            b2s[rep] = set(block)
            for s in block:
                s2b[s] = rep
        return cls(state_to_block=s2b, block_to_states=b2s)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class DifferentialTestEngine:
    """Compare an original LTS with its quotient at multiple levels.

    Usage::

        engine = DifferentialTestEngine(original_snapshot, quotient_snapshot, mapping)
        report = engine.run_full()
    """

    def __init__(
        self,
        original: LTSSnapshot,
        quotient: LTSSnapshot,
        mapping: Optional[QuotientMapping] = None,
        checker: Optional[PropertyChecker] = None,
        rng_seed: int = 42,
    ) -> None:
        self._orig = original
        self._quot = quotient
        self._mapping = mapping
        self._checker = checker
        self._rng = random.Random(rng_seed)
        self._tested_orig_states: Set[str] = set()
        self._tested_orig_transitions: Set[Tuple[str, str, str]] = set()
        self._tested_quot_states: Set[str] = set()
        self._tested_quot_transitions: Set[Tuple[str, str, str]] = set()

    # -- state comparison ----------------------------------------------------

    def compare_states(self) -> List[StateDiff]:
        """Compare reachable states via the quotient mapping."""
        diffs: List[StateDiff] = []
        if self._mapping is None:
            # Without mapping, just compare sets
            only_orig = self._orig.states - self._quot.states
            only_quot = self._quot.states - self._orig.states
            for s in only_orig:
                diffs.append(StateDiff(
                    state=s, in_original=True, in_quotient=False,
                    original_labels=self._orig.labels.get(s, frozenset()),
                    bug_kind=BugKind.MISSING_STATE,
                ))
            for s in only_quot:
                diffs.append(StateDiff(
                    state=s, in_original=False, in_quotient=True,
                    quotient_labels=self._quot.labels.get(s, frozenset()),
                    bug_kind=BugKind.EXTRA_STATE,
                ))
            self._tested_orig_states = self._orig.states
            self._tested_quot_states = self._quot.states
            return diffs

        # With mapping, verify every original state maps to a quotient block
        for state in self._orig.states:
            self._tested_orig_states.add(state)
            block = self._mapping.map_state(state)
            self._tested_quot_states.add(block)
            if block not in self._quot.states:
                diffs.append(StateDiff(
                    state=state, in_original=True, in_quotient=False,
                    original_labels=self._orig.labels.get(state, frozenset()),
                    bug_kind=BugKind.MISSING_STATE,
                ))
            else:
                # Check label consistency
                o_labels = self._orig.labels.get(state, frozenset())
                q_labels = self._quot.labels.get(block, frozenset())
                if o_labels and q_labels and o_labels != q_labels:
                    diffs.append(StateDiff(
                        state=state, in_original=True, in_quotient=True,
                        original_labels=o_labels, quotient_labels=q_labels,
                        bug_kind=BugKind.LABEL_MISMATCH,
                    ))
        return diffs

    # -- transition comparison -----------------------------------------------

    def compare_transitions(self) -> List[TransitionDiff]:
        """Compare transitions, mapping through the quotient if available."""
        diffs: List[TransitionDiff] = []

        orig_triples: Set[Tuple[str, str, str]] = set()
        for src, acts in self._orig.transitions.items():
            for act, dsts in acts.items():
                for d in dsts:
                    orig_triples.add((src, act, d))

        quot_triples: Set[Tuple[str, str, str]] = set()
        for src, acts in self._quot.transitions.items():
            for act, dsts in acts.items():
                for d in dsts:
                    quot_triples.add((src, act, d))

        if self._mapping is not None:
            # Map original transitions to quotient space
            mapped_orig: Set[Tuple[str, str, str]] = set()
            for src, act, dst in orig_triples:
                ms = self._mapping.map_state(src)
                md = self._mapping.map_state(dst)
                mapped_orig.add((ms, act, md))
                self._tested_orig_transitions.add((src, act, dst))

            # Transitions in original but not quotient
            for src, act, dst in mapped_orig - quot_triples:
                diffs.append(TransitionDiff(
                    src=src, action=act, dst=dst,
                    in_original=True, in_quotient=False,
                    bug_kind=BugKind.MISSING_TRANSITION,
                ))
            # Extra transitions in quotient
            for src, act, dst in quot_triples - mapped_orig:
                diffs.append(TransitionDiff(
                    src=src, action=act, dst=dst,
                    in_original=False, in_quotient=True,
                    bug_kind=BugKind.EXTRA_TRANSITION,
                ))
                self._tested_quot_transitions.add((src, act, dst))
        else:
            self._tested_orig_transitions = orig_triples
            self._tested_quot_transitions = quot_triples
            for t in orig_triples - quot_triples:
                diffs.append(TransitionDiff(
                    src=t[0], action=t[1], dst=t[2],
                    in_original=True, in_quotient=False,
                    bug_kind=BugKind.MISSING_TRANSITION,
                ))
            for t in quot_triples - orig_triples:
                diffs.append(TransitionDiff(
                    src=t[0], action=t[1], dst=t[2],
                    in_original=False, in_quotient=True,
                    bug_kind=BugKind.EXTRA_TRANSITION,
                ))
        return diffs

    # -- random walk comparison ----------------------------------------------

    def _walk(
        self, snapshot: LTSSnapshot, length: int
    ) -> List[str]:
        """Execute a random walk on the given LTS."""
        trace: List[str] = [snapshot.initial]
        current = snapshot.initial
        for _ in range(length):
            acts = snapshot.transitions.get(current, {})
            if not acts:
                break
            act = self._rng.choice(list(acts.keys()))
            dsts = acts[act]
            if not dsts:
                break
            nxt = self._rng.choice(list(dsts))
            trace.append(nxt)
            current = nxt
        return trace

    def random_walk_compare(
        self, walk_count: int = 100, walk_length: int = 50
    ) -> List[RandomWalkResult]:
        """Run random walks on both systems and compare traces."""
        results: List[RandomWalkResult] = []
        saved_state = self._rng.getstate()

        for _ in range(walk_count):
            seed = self._rng.randint(0, 2**31)
            # Walk on original
            self._rng.seed(seed)
            orig_trace = self._walk(self._orig, walk_length)
            # Walk on quotient with same decisions
            self._rng.seed(seed)
            quot_trace = self._walk(self._quot, walk_length)

            # Record coverage
            for s in orig_trace:
                self._tested_orig_states.add(s)
            for s in quot_trace:
                self._tested_quot_states.add(s)

            # Compare (via mapping if available)
            div_step = -1
            div_detail = ""
            min_len = min(len(orig_trace), len(quot_trace))
            for i in range(min_len):
                o_state = orig_trace[i]
                q_state = quot_trace[i]
                if self._mapping:
                    mapped = self._mapping.map_state(o_state)
                    if mapped != q_state:
                        div_step = i
                        div_detail = (
                            f"Step {i}: orig={o_state} (mapped={mapped}), "
                            f"quot={q_state}"
                        )
                        break
                else:
                    if o_state != q_state:
                        div_step = i
                        div_detail = f"Step {i}: orig={o_state}, quot={q_state}"
                        break

            if div_step < 0 and len(orig_trace) != len(quot_trace):
                div_step = min_len
                div_detail = (
                    f"Trace length mismatch: orig={len(orig_trace)}, "
                    f"quot={len(quot_trace)}"
                )

            results.append(RandomWalkResult(
                walk_length=walk_length,
                original_trace=orig_trace,
                quotient_trace=quot_trace,
                divergence_step=div_step,
                divergence_detail=div_detail,
            ))

        self._rng.setstate(saved_state)
        return results

    # -- property comparison -------------------------------------------------

    def compare_properties(
        self,
        properties: Sequence[PropertySpec],
        original_system: Any,
        quotient_system: Any,
        timeout: float = 60.0,
    ) -> List[Discrepancy]:
        """Check properties on both systems and return discrepancies."""
        if self._checker is None:
            return []
        discs: List[Discrepancy] = []
        for prop in properties:
            o_res = self._checker.check(original_system, prop, timeout)
            q_res = self._checker.check(quotient_system, prop, timeout)
            if o_res.verdict != q_res.verdict:
                if o_res.verdict == Verdict.VIOLATED and q_res.verdict == Verdict.SATISFIED:
                    kind = DiscrepancyKind.FALSE_POSITIVE
                elif o_res.verdict == Verdict.SATISFIED and q_res.verdict == Verdict.VIOLATED:
                    kind = DiscrepancyKind.FALSE_NEGATIVE
                else:
                    kind = DiscrepancyKind.TIMEOUT_MISMATCH
                discs.append(Discrepancy(
                    property_spec=prop, kind=kind,
                    original_verdict=o_res.verdict,
                    quotient_verdict=q_res.verdict,
                    details=f"Property: {prop.formula}",
                ))
        return discs

    # -- coverage analysis ---------------------------------------------------

    def _compute_coverage(self) -> CoverageInfo:
        orig_trans_total = sum(
            len(dsts)
            for acts in self._orig.transitions.values()
            for dsts in acts.values()
        )
        quot_trans_total = sum(
            len(dsts)
            for acts in self._quot.transitions.values()
            for dsts in acts.values()
        )
        return CoverageInfo(
            total_original_states=len(self._orig.states),
            tested_original_states=len(self._tested_orig_states),
            total_original_transitions=orig_trans_total,
            tested_original_transitions=len(self._tested_orig_transitions),
            total_quotient_states=len(self._quot.states),
            tested_quotient_states=len(self._tested_quot_states),
            total_quotient_transitions=quot_trans_total,
            tested_quotient_transitions=len(self._tested_quot_transitions),
        )

    # -- bug classification --------------------------------------------------

    def _classify_bugs(
        self,
        state_diffs: List[StateDiff],
        trans_diffs: List[TransitionDiff],
    ) -> List[Dict[str, Any]]:
        """Classify and aggregate bugs by kind."""
        counts: Dict[str, int] = {}
        examples: Dict[str, List[Dict[str, Any]]] = {}
        for sd in state_diffs:
            if sd.bug_kind:
                k = sd.bug_kind.name
                counts[k] = counts.get(k, 0) + 1
                examples.setdefault(k, []).append(sd.to_dict())
        for td in trans_diffs:
            if td.bug_kind:
                k = td.bug_kind.name
                counts[k] = counts.get(k, 0) + 1
                examples.setdefault(k, []).append(td.to_dict())

        # Check for hash collisions: states with same hash but different labels
        if self._mapping:
            for block, members in self._mapping.block_to_states.items():
                labels_in_block: Set[FrozenSet[str]] = set()
                for m in members:
                    lbl = self._orig.labels.get(m, frozenset())
                    labels_in_block.add(lbl)
                if len(labels_in_block) > 1:
                    k = BugKind.HASH_COLLISION.name
                    counts[k] = counts.get(k, 0) + 1
                    examples.setdefault(k, []).append({
                        "block": block,
                        "members": sorted(members),
                        "distinct_label_sets": len(labels_in_block),
                    })

        return [
            {"kind": k, "count": c, "examples": examples.get(k, [])[:5]}
            for k, c in sorted(counts.items())
        ]

    # -- full run ------------------------------------------------------------

    def run_full(
        self,
        walk_count: int = 100,
        walk_length: int = 50,
        properties: Optional[Sequence[PropertySpec]] = None,
        original_system: Any = None,
        quotient_system: Any = None,
    ) -> DifferentialTestReport:
        """Execute all differential tests and return a report."""
        t0 = time.monotonic()
        report = DifferentialTestReport()

        report.state_diffs = self.compare_states()
        report.transition_diffs = self.compare_transitions()
        report.random_walk_results = self.random_walk_compare(walk_count, walk_length)

        if properties and original_system and quotient_system:
            report.property_discrepancies = self.compare_properties(
                properties, original_system, quotient_system,
            )

        report.coverage = self._compute_coverage()
        report.bugs_found = self._classify_bugs(
            report.state_diffs, report.transition_diffs,
        )
        report.elapsed_seconds = time.monotonic() - t0
        return report
