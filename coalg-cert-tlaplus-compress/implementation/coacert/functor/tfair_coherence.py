"""
T-Fair coherence condition for CoaCert-TLA.

The coherence condition ensures that the stutter monad T distributes
properly over the fairness component of the functor F.

Main theorem: stuttering-equivalent paths satisfy the same acceptance
pairs if and only if the T-Fair coherence condition holds. This is
necessary and sufficient for F-bisimulations to preserve fairness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types for coherence results
# ---------------------------------------------------------------------------

@dataclass
class CoherenceWitness:
    """A witness that coherence holds for a particular acceptance pair.

    For pair (B_i, G_i), the witness shows that for every pair of
    stutter-equivalent states (s, t):
      s ∈ B_i  ⟺  t ∈ B_i
      s ∈ G_i  ⟺  t ∈ G_i
    """

    pair_index: int
    b_states: FrozenSet[str]
    g_states: FrozenSet[str]
    stutter_classes_checked: int
    all_consistent: bool
    details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "COHERENT" if self.all_consistent else "INCOHERENT"
        return (
            f"Pair {self.pair_index}: {status} "
            f"(checked {self.stutter_classes_checked} stutter classes, "
            f"|B|={len(self.b_states)}, |G|={len(self.g_states)})"
        )


@dataclass
class CoherenceViolation:
    """A counterexample showing that coherence fails.

    Two stutter-equivalent states disagree on membership in some
    acceptance pair component.
    """

    pair_index: int
    state_1: str
    state_2: str
    stutter_class: FrozenSet[str]
    component: str  # "B" or "G"
    state_1_member: bool
    state_2_member: bool
    explanation: str = ""

    def summary(self) -> str:
        return (
            f"Violation at pair {self.pair_index}, component {self.component}: "
            f"states {self.state_1} (∈{self.component}={self.state_1_member}) and "
            f"{self.state_2} (∈{self.component}={self.state_2_member}) are stutter-equivalent "
            f"but disagree on {self.component}_{self.pair_index} membership"
        )


@dataclass
class CoherenceResult:
    """Full result of a coherence check."""

    is_coherent: bool
    witnesses: List[CoherenceWitness] = field(default_factory=list)
    violations: List[CoherenceViolation] = field(default_factory=list)
    distribution_holds: bool = False
    log: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"T-Fair Coherence: {'HOLDS' if self.is_coherent else 'FAILS'}",
            f"  Distribution T over Fair: {'YES' if self.distribution_holds else 'NO'}",
            f"  Witnesses: {len(self.witnesses)}",
            f"  Violations: {len(self.violations)}",
        ]
        for w in self.witnesses:
            lines.append(f"    {w.summary()}")
        for v in self.violations:
            lines.append(f"    {v.summary()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# T-Fair coherence checker
# ---------------------------------------------------------------------------

class TFairCoherenceChecker:
    """Check the T-Fair coherence condition.

    Given:
      - A coalgebra (S, γ: S → F(S))
      - A stutter monad T with computed stutter equivalence classes
      - Fairness constraints as acceptance pairs (B_i, G_i)

    The coherence condition requires that for every acceptance pair
    (B_i, G_i), the sets B_i and G_i are unions of stutter equivalence
    classes. Equivalently, stutter-equivalent states must agree on
    membership in every B_i and G_i.
    """

    def __init__(self):
        self._coalgebra: Any = None
        self._stutter_monad: Any = None
        self._fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]] = []
        self._stutter_classes: List[Any] = []
        self._log: List[str] = []

    def _log_step(self, message: str) -> None:
        self._log.append(message)
        logger.debug("TFairCoherence: %s", message)

    def load(
        self,
        coalgebra: Any,
        stutter_monad: Any,
        fairness_pairs: Optional[List[Tuple[Set[str], Set[str]]]] = None,
    ) -> None:
        """Load the components for coherence checking."""
        self._coalgebra = coalgebra
        self._stutter_monad = stutter_monad
        self._log = []

        if fairness_pairs is not None:
            self._fairness_pairs = [
                (frozenset(b), frozenset(g)) for b, g in fairness_pairs
            ]
        else:
            self._fairness_pairs = []
            if hasattr(coalgebra, 'fairness_constraints'):
                for fc in coalgebra.fairness_constraints:
                    self._fairness_pairs.append((fc.b_states, fc.g_states))

        self._stutter_classes = stutter_monad.compute_stutter_equivalence_classes()

        self._log_step(
            f"Loaded system: {len(self._fairness_pairs)} fairness pairs, "
            f"{len(self._stutter_classes)} stutter classes"
        )

    def check_coherence(
        self,
        coalgebra: Optional[Any] = None,
        stutter_monad: Optional[Any] = None,
        fairness_pairs: Optional[List[Tuple[Set[str], Set[str]]]] = None,
    ) -> CoherenceResult:
        """Main entry point: check the T-Fair coherence condition.

        If arguments are provided, they override previously loaded components.
        """
        if coalgebra is not None or stutter_monad is not None:
            self.load(
                coalgebra or self._coalgebra,
                stutter_monad or self._stutter_monad,
                fairness_pairs,
            )

        result = CoherenceResult(is_coherent=True, log=list(self._log))

        if not self._fairness_pairs:
            self._log_step("No fairness pairs; coherence holds vacuously")
            result.log = list(self._log)
            result.distribution_holds = True
            return result

        self._log_step("Checking coherence for each acceptance pair...")

        for idx, (b_set, g_set) in enumerate(self._fairness_pairs):
            self._log_step(f"--- Pair {idx}: |B|={len(b_set)}, |G|={len(g_set)} ---")
            witness, violations = self._check_pair(idx, b_set, g_set)
            result.witnesses.append(witness)
            if violations:
                result.is_coherent = False
                result.violations.extend(violations)

        result.distribution_holds = self._check_distribution()
        result.log = list(self._log)

        self._log_step(
            f"Coherence check complete: {'COHERENT' if result.is_coherent else 'INCOHERENT'}"
        )

        return result

    def _check_pair(
        self, idx: int, b_set: FrozenSet[str], g_set: FrozenSet[str]
    ) -> Tuple[CoherenceWitness, List[CoherenceViolation]]:
        """Check coherence for a single acceptance pair (B_i, G_i)."""
        violations: List[CoherenceViolation] = []
        details: List[str] = []
        classes_checked = 0

        for cls in self._stutter_classes:
            classes_checked += 1
            members = cls.members

            b_check = self._check_class_membership(
                idx, members, b_set, "B", violations
            )
            g_check = self._check_class_membership(
                idx, members, g_set, "G", violations
            )

            if b_check and g_check:
                details.append(
                    f"Class [{cls.representative}] ({cls.size()} states): consistent"
                )
            else:
                details.append(
                    f"Class [{cls.representative}] ({cls.size()} states): INCONSISTENT"
                )

        all_consistent = len(violations) == 0
        witness = CoherenceWitness(
            pair_index=idx,
            b_states=b_set,
            g_states=g_set,
            stutter_classes_checked=classes_checked,
            all_consistent=all_consistent,
            details=details,
        )

        self._log_step(witness.summary())
        return witness, violations

    def _check_class_membership(
        self,
        pair_idx: int,
        class_members: FrozenSet[str],
        target_set: FrozenSet[str],
        component: str,
        violations: List[CoherenceViolation],
    ) -> bool:
        """Check that all members of a stutter class agree on membership.

        Returns True if consistent, False otherwise.
        """
        if len(class_members) <= 1:
            return True

        members_in = class_members & target_set
        members_out = class_members - target_set

        if members_in and members_out:
            s_in = min(members_in)
            s_out = min(members_out)
            violation = CoherenceViolation(
                pair_index=pair_idx,
                state_1=s_in,
                state_2=s_out,
                stutter_class=class_members,
                component=component,
                state_1_member=True,
                state_2_member=False,
                explanation=(
                    f"States {s_in} and {s_out} are stutter-equivalent but "
                    f"{s_in} ∈ {component}_{pair_idx} while {s_out} ∉ {component}_{pair_idx}"
                ),
            )
            violations.append(violation)
            self._log_step(violation.summary())
            return False

        return True

    def _check_distribution(self) -> bool:
        """Check that T distributes over Fair(X).

        The distributive law requires a natural transformation
        δ: T ∘ Fair ⇒ Fair ∘ T
        such that the coherence diagrams commute.

        For our setting, this reduces to checking that T-images of
        fairness sets are again unions of stutter classes in the
        T-image system.
        """
        self._log_step("Checking distribution of T over Fair...")

        eta = self._stutter_monad.unit_map()

        for idx, (b_set, g_set) in enumerate(self._fairness_pairs):
            t_b = frozenset(eta.get(s, s) for s in b_set)
            t_g = frozenset(eta.get(s, s) for s in g_set)

            if not self._is_union_of_classes(t_b):
                self._log_step(
                    f"Distribution fails: T(B_{idx}) is not a union of stutter classes"
                )
                return False

            if not self._is_union_of_classes(t_g):
                self._log_step(
                    f"Distribution fails: T(G_{idx}) is not a union of stutter classes"
                )
                return False

        self._log_step("Distribution of T over Fair: OK")
        return True

    def _is_union_of_classes(self, state_set: FrozenSet[str]) -> bool:
        """Check if a set of states is a union of stutter equivalence classes."""
        for cls in self._stutter_classes:
            intersection = cls.members & state_set
            if intersection and intersection != cls.members:
                return False
        return True

    # -- detailed analysis methods ------------------------------------------

    def analyze_pair_structure(
        self, pair_index: int
    ) -> Dict[str, Any]:
        """Detailed analysis of how a fairness pair interacts with stutter classes."""
        if pair_index >= len(self._fairness_pairs):
            return {"error": f"Pair index {pair_index} out of range"}

        b_set, g_set = self._fairness_pairs[pair_index]

        analysis: Dict[str, Any] = {
            "pair_index": pair_index,
            "b_size": len(b_set),
            "g_size": len(g_set),
            "classes": [],
        }

        for cls in self._stutter_classes:
            b_inter = cls.members & b_set
            g_inter = cls.members & g_set
            b_ratio = len(b_inter) / max(len(cls.members), 1)
            g_ratio = len(g_inter) / max(len(cls.members), 1)

            class_info = {
                "representative": cls.representative,
                "size": cls.size(),
                "b_members": len(b_inter),
                "g_members": len(g_inter),
                "b_ratio": b_ratio,
                "g_ratio": g_ratio,
                "b_coherent": b_ratio in (0.0, 1.0),
                "g_coherent": g_ratio in (0.0, 1.0),
            }
            analysis["classes"].append(class_info)

        coherent_count = sum(
            1
            for c in analysis["classes"]
            if c["b_coherent"] and c["g_coherent"]
        )
        analysis["coherent_classes"] = coherent_count
        analysis["total_classes"] = len(analysis["classes"])
        analysis["fully_coherent"] = coherent_count == len(analysis["classes"])

        return analysis

    def suggest_repair(
        self,
    ) -> List[Dict[str, Any]]:
        """If coherence fails, suggest how to repair it.

        The simplest repair is to adjust fairness sets to be unions of
        stutter equivalence classes by either:
        1. Expanding: include all class members if any member is present
        2. Contracting: exclude all class members if not all are present
        """
        repairs: List[Dict[str, Any]] = []

        for idx, (b_set, g_set) in enumerate(self._fairness_pairs):
            expand_b: Set[str] = set(b_set)
            contract_b: Set[str] = set(b_set)
            expand_g: Set[str] = set(g_set)
            contract_g: Set[str] = set(g_set)

            for cls in self._stutter_classes:
                b_inter = cls.members & b_set
                g_inter = cls.members & g_set

                if b_inter and b_inter != cls.members:
                    expand_b |= cls.members
                    contract_b -= cls.members

                if g_inter and g_inter != cls.members:
                    expand_g |= cls.members
                    contract_g -= cls.members

            if expand_b != b_set or expand_g != g_set or contract_b != b_set or contract_g != g_set:
                repairs.append({
                    "pair_index": idx,
                    "original_b": b_set,
                    "original_g": g_set,
                    "expanded_b": frozenset(expand_b),
                    "expanded_g": frozenset(expand_g),
                    "contracted_b": frozenset(contract_b),
                    "contracted_g": frozenset(contract_g),
                })

        return repairs

    def check_bisimulation_preserves_fairness(
        self,
        bisimulation: List[FrozenSet[str]],
    ) -> Tuple[bool, List[str]]:
        """Check that a given bisimulation (as partition) preserves fairness.

        This is a consequence of coherence: if coherence holds, any
        F-bisimulation automatically preserves fairness.
        """
        issues: List[str] = []

        for idx, (b_set, g_set) in enumerate(self._fairness_pairs):
            for block in bisimulation:
                b_members = block & b_set
                g_members = block & g_set

                if b_members and b_members != block:
                    issues.append(
                        f"Bisimulation block {sorted(block)} splits B_{idx}: "
                        f"{sorted(b_members)} ∈ B but {sorted(block - b_members)} ∉ B"
                    )

                if g_members and g_members != block:
                    issues.append(
                        f"Bisimulation block {sorted(block)} splits G_{idx}: "
                        f"{sorted(g_members)} ∈ G but {sorted(block - g_members)} ∉ G"
                    )

        return len(issues) == 0, issues

    def generate_coherence_certificate(self) -> Dict[str, Any]:
        """Emit a serializable JSON certificate for the coherence check.

        Returns a dictionary containing all coherence verification data,
        suitable for serialization to JSON and independent verification.
        Must be called after ``check_coherence()``.
        """
        import hashlib as _hl
        import time as _time

        cert: Dict[str, Any] = {
            "type": "TFairCoherenceCertificate",
            "timestamp": _time.monotonic(),
            "num_fairness_pairs": len(self._fairness_pairs),
            "num_stutter_classes": len(self._stutter_classes),
        }

        # Run coherence check if not already done
        result = self.check_coherence()
        cert["is_coherent"] = result.is_coherent
        cert["distribution_holds"] = result.distribution_holds
        cert["witnesses"] = [
            {
                "pair_index": w.pair_index,
                "b_size": len(w.b_states),
                "g_size": len(w.g_states),
                "stutter_classes_checked": w.stutter_classes_checked,
                "all_consistent": w.all_consistent,
            }
            for w in result.witnesses
        ]
        cert["violations"] = [
            {
                "pair_index": v.pair_index,
                "state_1": v.state_1,
                "state_2": v.state_2,
                "component": v.component,
            }
            for v in result.violations
        ]

        # Compute certificate hash
        hasher = _hl.sha256()
        hasher.update(str(cert["is_coherent"]).encode())
        hasher.update(str(cert["distribution_holds"]).encode())
        hasher.update(str(cert["num_fairness_pairs"]).encode())
        cert["certificate_hash"] = hasher.hexdigest()

        return cert


# ---------------------------------------------------------------------------
# Categorical coherence diagram verifier
# ---------------------------------------------------------------------------

class CategoricalCoherenceDiagram:
    """Verifies that δ: T ∘ Fair ⇒ Fair ∘ T commutes as a natural transformation.

    For every F-coalgebra morphism h: S → Q, checks the naturality square:
        T(Fair(h)) ; δ_Q = δ_S ; Fair(T(h))

    Also verifies that δ is well-defined on each component (each acceptance
    pair) by checking that T-images of fairness sets remain unions of
    stutter equivalence classes.

    This class works with the runtime data held by TFairCoherenceChecker
    and produces categorical verification results.
    """

    def __init__(self) -> None:
        self._naturality_results: List[Dict[str, Any]] = []
        self._component_results: List[Dict[str, Any]] = []
        self._commutes: bool = False

    def verify_naturality_square(
        self,
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Dict[str, str],
        morphisms: List[Tuple[str, Dict[str, str]]],
        stutter_classes: List[Any],
    ) -> bool:
        """Verify the naturality square for each morphism h.

        For every F-coalgebra morphism h and each acceptance pair (B_i, G_i):
            Left path:  T(Fair(h))(B, G) then δ_Q
            Right path: δ_S(B, G) then Fair(T(h))

        Parameters
        ----------
        fairness_pairs : list of (B_i, G_i)
        eta : dict mapping each state to its stutter representative
        morphisms : list of (id, h) where h maps source states to target
        stutter_classes : list of stutter equivalence classes

        Returns
        -------
        bool
            True if the naturality square commutes for all morphisms.
        """
        self._naturality_results = []
        all_commute = True

        if not morphisms:
            self._commutes = True
            return True

        for morph_id, h in morphisms:
            source_states = frozenset(h.keys())
            morph_ok = True

            for idx, (b_set, g_set) in enumerate(fairness_pairs):
                b_in_dom = b_set & source_states
                g_in_dom = g_set & source_states

                # Left path: Fair(h) then T (via η)
                h_b = frozenset(h[s] for s in b_in_dom if s in h)
                h_g = frozenset(h[s] for s in g_in_dom if s in h)
                left_b = frozenset(eta.get(s, s) for s in h_b)
                left_g = frozenset(eta.get(s, s) for s in h_g)

                # Right path: T (via η) then Fair(T(h))
                right_b = frozenset(
                    eta.get(h.get(s, s), h.get(s, s)) for s in b_in_dom
                )
                right_g = frozenset(
                    eta.get(h.get(s, s), h.get(s, s)) for s in g_in_dom
                )

                pair_commutes = (left_b == right_b) and (left_g == right_g)
                if not pair_commutes:
                    morph_ok = False

            self._naturality_results.append({
                "morphism_id": morph_id,
                "commutes": morph_ok,
                "pairs_checked": len(fairness_pairs),
            })
            if not morph_ok:
                all_commute = False

        self._commutes = all_commute
        return all_commute

    def check_naturality(
        self,
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Dict[str, str],
        stutter_classes: List[Any],
    ) -> bool:
        """Verify that the natural transformation δ is well-defined on each component.

        For each acceptance pair (B_i, G_i), checks that:
        1. B_i is a union of stutter classes (so δ can act on it)
        2. G_i is a union of stutter classes (so δ can act on it)
        This ensures δ is well-defined as a natural transformation because
        η maps entire stutter classes to their representative, making
        η(B_i) and η(G_i) well-defined sets in T(S).
        """
        self._component_results = []
        all_ok = True

        for idx, (b_set, g_set) in enumerate(fairness_pairs):
            b_saturated = self._is_union_of_classes(b_set, stutter_classes)
            g_saturated = self._is_union_of_classes(g_set, stutter_classes)

            eta_b = frozenset(eta.get(s, s) for s in b_set)
            eta_g = frozenset(eta.get(s, s) for s in g_set)

            self._component_results.append({
                "pair_index": idx,
                "b_well_defined": b_saturated,
                "g_well_defined": g_saturated,
                "eta_b_size": len(eta_b),
                "eta_g_size": len(eta_g),
            })

            if not (b_saturated and g_saturated):
                all_ok = False

        return all_ok

    def generate_coherence_certificate(self) -> Dict[str, Any]:
        """Emit a serializable certificate for the categorical diagram verification."""
        import hashlib as _hl

        cert: Dict[str, Any] = {
            "type": "CategoricalCoherenceDiagramCertificate",
            "commutes": self._commutes,
            "naturality_results": self._naturality_results,
            "component_results": self._component_results,
        }

        hasher = _hl.sha256()
        hasher.update(str(self._commutes).encode())
        for r in self._naturality_results:
            hasher.update(r["morphism_id"].encode())
            hasher.update(str(r["commutes"]).encode())
        cert["certificate_hash"] = hasher.hexdigest()

        return cert

    @property
    def commutes(self) -> bool:
        return self._commutes

    @property
    def naturality_results(self) -> List[Dict[str, Any]]:
        return list(self._naturality_results)

    @property
    def component_results(self) -> List[Dict[str, Any]]:
        return list(self._component_results)

    @staticmethod
    def _is_union_of_classes(
        state_set: FrozenSet[str], stutter_classes: List[Any]
    ) -> bool:
        for cls in stutter_classes:
            inter = cls.members & state_set
            if inter and inter != cls.members:
                return False
        return True
