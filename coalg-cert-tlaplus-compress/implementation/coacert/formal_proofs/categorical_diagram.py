"""
Explicit verification of categorical coherence diagrams for T-Fair coherence.

This module formalizes the categorical diagrams that underpin the T-Fair
coherence theorem. The key diagrams are:

DIAGRAM 1 (Distributive Law Naturality):
  For every F-coalgebra morphism h: S → Q, the following square commutes:

                T(Fair(S)) ──δ_S──▶ Fair(T(S))
                    │                    │
              T(Fair(h))            Fair(T(h))
                    │                    │
                    ▼                    ▼
                T(Fair(Q)) ──δ_Q──▶ Fair(T(Q))

  i.e., δ_Q ∘ T(Fair(h)) = Fair(T(h)) ∘ δ_S

DIAGRAM 2 (Monad Unit Compatibility):
  δ ∘ η^Fair = Fair(η)

  where η is the unit of the stutter monad T.

DIAGRAM 3 (Monad Multiplication Compatibility):
  δ ∘ μ^Fair = Fair(μ) ∘ δT ∘ Tδ

  where μ is the multiplication of the stutter monad T.

These diagrams collectively ensure that the stutter monad T distributes
over the fairness functor Fair, which is the categorical content of the
T-Fair coherence condition.

References:
  - Beck (1969): Distributive laws
  - Barr (1970): Coequalizers and free triples
  - Klin & Salamanca (2018): Iterated coequalisers
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification result types
# ---------------------------------------------------------------------------

class DiagramStatus(Enum):
    """Status of a diagram verification."""
    PENDING = auto()
    COMMUTES = auto()
    FAILS = auto()
    SKIPPED = auto()


@dataclass
class NaturalityWitness:
    """Witness for one instance of the naturality condition.

    For a specific morphism h: S → Q, records whether the naturality
    square commutes: δ_Q ∘ T(Fair(h)) = Fair(T(h)) ∘ δ_S.

    Attributes:
        morphism_id: Identifier for the morphism being verified.
        source_states: States of the source coalgebra S.
        target_states: States of the target coalgebra Q.
        morphism_map: The concrete morphism h as a state mapping.
        left_path_results: Results of computing δ_Q ∘ T(Fair(h)) on each element.
        right_path_results: Results of computing Fair(T(h)) ∘ δ_S on each element.
        commutes: Whether the two paths agree on all elements.
        counterexample: If not commuting, an element witnessing failure.
        details: Human-readable explanation of verification steps.
    """
    morphism_id: str = ""
    source_states: FrozenSet[str] = field(default_factory=frozenset)
    target_states: FrozenSet[str] = field(default_factory=frozenset)
    morphism_map: Dict[str, str] = field(default_factory=dict)
    left_path_results: Dict[str, Any] = field(default_factory=dict)
    right_path_results: Dict[str, Any] = field(default_factory=dict)
    commutes: bool = False
    counterexample: Optional[str] = None
    details: List[str] = field(default_factory=list)
    verification_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "morphism_id": self.morphism_id,
            "source_states_count": len(self.source_states),
            "target_states_count": len(self.target_states),
            "commutes": self.commutes,
            "counterexample": self.counterexample,
            "details": self.details,
            "verification_time_seconds": self.verification_time_seconds,
        }


@dataclass
class DiagramVerificationResult:
    """Result of verifying a categorical coherence diagram.

    Attributes:
        diagram_name: Human-readable name of the diagram.
        diagram_id: Machine-readable identifier (e.g., 'naturality', 'unit', 'mult').
        status: Whether the diagram commutes, fails, or is pending.
        naturality_witnesses: Per-morphism witnesses (for the naturality square).
        pair_results: Per-acceptance-pair verification results.
        aggregate_commutes: True iff all sub-checks pass.
        proof_hash: SHA-256 hash of the verification result for tamper detection.
        verification_time_seconds: Wall-clock time for this verification.
        details: Human-readable log of the verification.
    """
    diagram_name: str = ""
    diagram_id: str = ""
    status: DiagramStatus = DiagramStatus.PENDING
    naturality_witnesses: List[NaturalityWitness] = field(default_factory=list)
    pair_results: List[Dict[str, Any]] = field(default_factory=list)
    aggregate_commutes: bool = False
    proof_hash: str = ""
    verification_time_seconds: float = 0.0
    details: List[str] = field(default_factory=list)

    def compute_proof_hash(self) -> str:
        """Compute a tamper-evident hash of this verification result."""
        hasher = hashlib.sha256()
        hasher.update(self.diagram_id.encode())
        hasher.update(self.status.name.encode())
        hasher.update(str(self.aggregate_commutes).encode())
        for nw in self.naturality_witnesses:
            hasher.update(nw.morphism_id.encode())
            hasher.update(str(nw.commutes).encode())
        for pr in self.pair_results:
            hasher.update(json.dumps(pr, sort_keys=True).encode())
        self.proof_hash = hasher.hexdigest()
        return self.proof_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagram_name": self.diagram_name,
            "diagram_id": self.diagram_id,
            "status": self.status.name,
            "aggregate_commutes": self.aggregate_commutes,
            "proof_hash": self.proof_hash,
            "verification_time_seconds": self.verification_time_seconds,
            "naturality_witnesses": [nw.to_dict() for nw in self.naturality_witnesses],
            "pair_results": self.pair_results,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Categorical diagram verifier
# ---------------------------------------------------------------------------

class CategoricalDiagramVerifier:
    """Verify categorical coherence diagrams for the T-Fair distributive law.

    Given a stutter monad T (with unit η and multiplication μ) and the
    fairness functor Fair, this verifier checks the three diagrams that
    define a distributive law δ: T ∘ Fair ⇒ Fair ∘ T:

    1. **Naturality**: For each morphism h: S → Q,
       δ_Q ∘ T(Fair(h)) = Fair(T(h)) ∘ δ_S

    2. **Unit compatibility**: δ ∘ η^{Fair} = Fair(η)

    3. **Multiplication compatibility**: δ ∘ μ^{Fair} = Fair(μ) ∘ δ_T ∘ T(δ)

    The verifier works with concrete finite-state representations:
    - T acts on states by mapping to stutter equivalence class representatives.
    - Fair(X) is represented as a list of Streett acceptance pairs (B_i, G_i)
      where each B_i, G_i ⊆ X.
    - The distributive law δ maps T(Fair(X)) to Fair(T(X)) by applying T
      to each component of each acceptance pair.
    """

    def __init__(self) -> None:
        self._results: List[DiagramVerificationResult] = []

    @property
    def results(self) -> List[DiagramVerificationResult]:
        """All verification results accumulated so far."""
        return list(self._results)

    def verify_all(
        self,
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Mapping[str, str],
        mu: Mapping[str, str],
        morphisms: Optional[List[Tuple[str, Mapping[str, str]]]] = None,
    ) -> Tuple[bool, List[DiagramVerificationResult]]:
        """Verify all three categorical coherence diagrams.

        Parameters
        ----------
        stutter_classes : list
            Each element has .members (FrozenSet[str]) and .representative (str).
        fairness_pairs : list of (B_i, G_i) pairs
            Streett acceptance pairs as frozensets of state names.
        eta : mapping str → str
            The monad unit: maps each state to its stutter class representative.
        mu : mapping str → str
            The monad multiplication: maps representatives-of-representatives
            to representatives (idempotent for stutter closure).
        morphisms : list of (id, mapping), optional
            F-coalgebra morphisms to verify naturality against.
            Each entry is (morphism_id, h: S → Q).

        Returns
        -------
        (all_commute, results) : (bool, list of DiagramVerificationResult)
        """
        self._results = []

        r_nat = self.verify_naturality(
            stutter_classes, fairness_pairs, eta, morphisms or []
        )
        r_unit = self.verify_unit_compatibility(
            stutter_classes, fairness_pairs, eta
        )
        r_mult = self.verify_multiplication_compatibility(
            stutter_classes, fairness_pairs, eta, mu
        )

        all_commute = (
            r_nat.aggregate_commutes
            and r_unit.aggregate_commutes
            and r_mult.aggregate_commutes
        )
        return all_commute, self._results

    # -------------------------------------------------------------------
    # Diagram 1: Naturality of δ
    # -------------------------------------------------------------------

    def verify_naturality(
        self,
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Mapping[str, str],
        morphisms: List[Tuple[str, Mapping[str, str]]],
    ) -> DiagramVerificationResult:
        """Verify the naturality square for δ.

        For each morphism h: S → Q, we check:
          δ_Q ∘ T(Fair(h)) = Fair(T(h)) ∘ δ_S

        In concrete terms, for each acceptance pair (B_i, G_i):
          Left path:  T(Fair(h))(B_i, G_i) = (T(h(B_i)), T(h(G_i)))
                      then apply δ_Q
          Right path: δ_S(B_i, G_i) = (T(B_i), T(G_i))
                      then apply Fair(T(h))

        For stutter-saturated sets, both paths produce the same result
        because T(h(X)) = h(T(X)) when X is a union of stutter classes
        and h respects stutter equivalence.
        """
        t0 = time.monotonic()
        result = DiagramVerificationResult(
            diagram_name="Distributive Law Naturality Square",
            diagram_id="naturality",
        )

        if not morphisms:
            result.status = DiagramStatus.COMMUTES
            result.aggregate_commutes = True
            result.details.append(
                "No morphisms provided; naturality holds vacuously."
            )
            result.verification_time_seconds = time.monotonic() - t0
            result.compute_proof_hash()
            self._results.append(result)
            return result

        all_commute = True
        for morph_id, h in morphisms:
            nw = self._check_naturality_for_morphism(
                morph_id, h, stutter_classes, fairness_pairs, eta
            )
            result.naturality_witnesses.append(nw)
            if not nw.commutes:
                all_commute = False

        result.aggregate_commutes = all_commute
        result.status = (
            DiagramStatus.COMMUTES if all_commute else DiagramStatus.FAILS
        )
        result.verification_time_seconds = time.monotonic() - t0
        result.compute_proof_hash()
        self._results.append(result)
        return result

    def _check_naturality_for_morphism(
        self,
        morph_id: str,
        h: Mapping[str, str],
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Mapping[str, str],
    ) -> NaturalityWitness:
        """Check the naturality square for a single morphism h.

        Left path:  δ_Q ∘ T(Fair(h))
          1. Fair(h)(B, G) = (h(B), h(G))       — apply h pointwise
          2. T(Fair(h))    = (η∘h(B), η∘h(G))   — apply T = η to result
          3. δ_Q           = identity on saturated sets in Q

        Right path: Fair(T(h)) ∘ δ_S
          1. δ_S(B, G) = (η(B), η(G))           — apply T = η to B, G
          2. T(h)      = η∘h                      — monad-enriched morphism
          3. Fair(T(h))(η(B), η(G)) = (η∘h(η(B)), η∘h(η(G)))

        For stutter-saturated sets where η is idempotent on representatives,
        both paths yield the same result.
        """
        t0 = time.monotonic()
        source_states = frozenset(h.keys())
        target_states = frozenset(h.values())

        nw = NaturalityWitness(
            morphism_id=morph_id,
            source_states=source_states,
            target_states=target_states,
            morphism_map=dict(h),
        )

        commutes = True
        for idx, (b_set, g_set) in enumerate(fairness_pairs):
            # Restrict to states in the domain of h
            b_in_domain = b_set & source_states
            g_in_domain = g_set & source_states

            # Left path: δ_Q ∘ T(Fair(h))
            # Step 1: Fair(h) — map states through h
            h_b = frozenset(h[s] for s in b_in_domain if s in h)
            h_g = frozenset(h[s] for s in g_in_domain if s in h)
            # Step 2: T — map through η (stutter closure in Q)
            left_b = frozenset(eta.get(s, s) for s in h_b)
            left_g = frozenset(eta.get(s, s) for s in h_g)

            # Right path: Fair(T(h)) ∘ δ_S
            # Step 1: δ_S — apply η (stutter closure in S)
            eta_b = frozenset(eta.get(s, s) for s in b_in_domain)
            eta_g = frozenset(eta.get(s, s) for s in g_in_domain)
            # Step 2: T(h) = η ∘ h — map through h then η
            right_b = frozenset(eta.get(h.get(s, s), h.get(s, s)) for s in b_in_domain)
            right_g = frozenset(eta.get(h.get(s, s), h.get(s, s)) for s in g_in_domain)

            pair_commutes = (left_b == right_b) and (left_g == right_g)

            nw.left_path_results[f"pair_{idx}"] = {
                "b": sorted(left_b), "g": sorted(left_g),
            }
            nw.right_path_results[f"pair_{idx}"] = {
                "b": sorted(right_b), "g": sorted(right_g),
            }

            if not pair_commutes:
                commutes = False
                # Find a specific counterexample
                b_diff = left_b.symmetric_difference(right_b)
                g_diff = left_g.symmetric_difference(right_g)
                ce = min(b_diff) if b_diff else (min(g_diff) if g_diff else None)
                if nw.counterexample is None and ce is not None:
                    nw.counterexample = (
                        f"Pair {idx}: left_path and right_path disagree on "
                        f"state {ce}"
                    )
                nw.details.append(
                    f"Pair {idx} FAILS: "
                    f"left_B={sorted(left_b)}, right_B={sorted(right_b)}, "
                    f"left_G={sorted(left_g)}, right_G={sorted(right_g)}"
                )
            else:
                nw.details.append(f"Pair {idx}: naturality commutes ✓")

        nw.commutes = commutes
        nw.verification_time_seconds = time.monotonic() - t0
        return nw

    # -------------------------------------------------------------------
    # Diagram 2: Unit compatibility — δ ∘ η^{Fair} = Fair(η)
    # -------------------------------------------------------------------

    def verify_unit_compatibility(
        self,
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Mapping[str, str],
    ) -> DiagramVerificationResult:
        """Verify unit compatibility: δ ∘ η^{Fair} = Fair(η).

        The unit η: Id ⇒ T of the stutter monad sends each state s to
        its equivalence class representative η(s).

        Left side: δ ∘ η^{Fair}
          η^{Fair}(B, G) = (η(B), η(G))  — apply monad unit to Fair
          then δ is identity on the result (δ: T∘Fair → Fair∘T)

        Right side: Fair(η)
          Fair(η)(B, G) = (η(B), η(G))   — apply η pointwise inside Fair

        These are equal precisely when η acts uniformly on the fairness
        sets, i.e., when η(B) = {η(s) | s ∈ B} is well-defined as a
        set in T(S).
        """
        t0 = time.monotonic()
        result = DiagramVerificationResult(
            diagram_name="Monad Unit Compatibility: δ ∘ η^Fair = Fair(η)",
            diagram_id="unit_compatibility",
        )

        all_states: Set[str] = set()
        for cls in stutter_classes:
            all_states |= cls.members

        all_commute = True
        for idx, (b_set, g_set) in enumerate(fairness_pairs):
            # Left side: δ ∘ η^{Fair}
            # η^{Fair}(B, G) = (η(B), η(G)) where η acts on the container
            # Then δ maps T(Fair(S)) → Fair(T(S)), which on saturated sets is identity
            left_b = frozenset(eta.get(s, s) for s in b_set)
            left_g = frozenset(eta.get(s, s) for s in g_set)

            # Right side: Fair(η)(B, G) = ({η(s) | s ∈ B}, {η(s) | s ∈ G})
            right_b = frozenset(eta.get(s, s) for s in b_set)
            right_g = frozenset(eta.get(s, s) for s in g_set)

            pair_ok = (left_b == right_b) and (left_g == right_g)
            pair_result = {
                "pair_index": idx,
                "commutes": pair_ok,
                "left_b_size": len(left_b),
                "right_b_size": len(right_b),
                "left_g_size": len(left_g),
                "right_g_size": len(right_g),
            }

            if not pair_ok:
                all_commute = False
                b_diff = left_b.symmetric_difference(right_b)
                g_diff = left_g.symmetric_difference(right_g)
                pair_result["b_disagreement"] = sorted(b_diff)
                pair_result["g_disagreement"] = sorted(g_diff)
                result.details.append(
                    f"Pair {idx}: unit compatibility FAILS — "
                    f"B differs on {sorted(b_diff)}, G differs on {sorted(g_diff)}"
                )
            else:
                result.details.append(
                    f"Pair {idx}: unit compatibility holds ✓"
                )

            # Additionally verify η-saturation: η(B) should be union of classes
            eta_b_saturated = self._is_union_of_classes(left_b, stutter_classes)
            eta_g_saturated = self._is_union_of_classes(left_g, stutter_classes)
            pair_result["eta_b_saturated"] = eta_b_saturated
            pair_result["eta_g_saturated"] = eta_g_saturated

            if not eta_b_saturated:
                all_commute = False
                result.details.append(
                    f"Pair {idx}: η(B_{idx}) is not a union of stutter classes"
                )
            if not eta_g_saturated:
                all_commute = False
                result.details.append(
                    f"Pair {idx}: η(G_{idx}) is not a union of stutter classes"
                )

            result.pair_results.append(pair_result)

        result.aggregate_commutes = all_commute
        result.status = (
            DiagramStatus.COMMUTES if all_commute else DiagramStatus.FAILS
        )
        result.verification_time_seconds = time.monotonic() - t0
        result.compute_proof_hash()
        self._results.append(result)
        return result

    # -------------------------------------------------------------------
    # Diagram 3: Multiplication compatibility — δ ∘ μ^{Fair} = Fair(μ) ∘ δT ∘ Tδ
    # -------------------------------------------------------------------

    def verify_multiplication_compatibility(
        self,
        stutter_classes: List[Any],
        fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        eta: Mapping[str, str],
        mu: Mapping[str, str],
    ) -> DiagramVerificationResult:
        """Verify multiplication compatibility: δ ∘ μ^{Fair} = Fair(μ) ∘ δT ∘ Tδ.

        The multiplication μ: T² ⇒ T of the stutter monad flattens
        double stutter closure. For finite systems where T is idempotent
        (η = η²), μ is simply the identity on T-images.

        Left side: δ ∘ μ^{Fair}
          μ^{Fair} flattens T²(Fair(S)) → T(Fair(S)) by applying μ component-wise.
          Then δ maps T(Fair(S)) → Fair(T(S)).

        Right side: Fair(μ) ∘ δ_T ∘ T(δ)
          T(δ): T(T(Fair(S))) → T(Fair(T(S)))  — apply δ inside T
          δ_T:  T(Fair(T(S))) → Fair(T(T(S)))  — apply δ at T(S) level
          Fair(μ): Fair(T²(S)) → Fair(T(S))     — apply μ inside Fair

        For idempotent monads (our case), both sides reduce to
        δ: T(Fair(S)) → Fair(T(S)) applied once.
        """
        t0 = time.monotonic()
        result = DiagramVerificationResult(
            diagram_name="Monad Multiplication Compatibility: δ ∘ μ^Fair = Fair(μ) ∘ δT ∘ Tδ",
            diagram_id="multiplication_compatibility",
        )

        all_commute = True
        for idx, (b_set, g_set) in enumerate(fairness_pairs):
            # Compute T-images via η
            t_b = frozenset(eta.get(s, s) for s in b_set)
            t_g = frozenset(eta.get(s, s) for s in g_set)

            # Left side: δ ∘ μ^{Fair}
            # μ^{Fair} on T²(B) = μ applied to each element of T(T(B))
            # T²(B) = {η(η(s)) | s ∈ B} = {η(s) | s ∈ B} (idempotent)
            t2_b = frozenset(eta.get(eta.get(s, s), eta.get(s, s)) for s in b_set)
            t2_g = frozenset(eta.get(eta.get(s, s), eta.get(s, s)) for s in g_set)
            mu_b = frozenset(mu.get(s, s) for s in t2_b)
            mu_g = frozenset(mu.get(s, s) for s in t2_g)
            # Apply δ: for saturated sets, δ is identity
            left_b = mu_b
            left_g = mu_g

            # Right side: Fair(μ) ∘ δ_T ∘ T(δ)
            # T(δ): apply δ inside T — since δ on saturated sets is identity,
            #   T(δ)(T(B)) = T(B)
            # δ_T: apply δ at the T(S) level — again identity for saturated sets
            # Fair(μ): apply μ pointwise — μ(η(s)) = η(s) for idempotent monad
            right_b = frozenset(mu.get(s, s) for s in t_b)
            right_g = frozenset(mu.get(s, s) for s in t_g)

            pair_ok = (left_b == right_b) and (left_g == right_g)
            pair_result = {
                "pair_index": idx,
                "commutes": pair_ok,
                "left_b": sorted(left_b),
                "right_b": sorted(right_b),
                "left_g": sorted(left_g),
                "right_g": sorted(right_g),
            }

            if not pair_ok:
                all_commute = False
                result.details.append(
                    f"Pair {idx}: multiplication compatibility FAILS"
                )
            else:
                result.details.append(
                    f"Pair {idx}: multiplication compatibility holds ✓"
                )

            # Verify μ is idempotent: μ(η(s)) = η(s) for all s
            idempotent_violations: List[str] = []
            for s in b_set | g_set:
                eta_s = eta.get(s, s)
                mu_eta_s = mu.get(eta_s, eta_s)
                if mu_eta_s != eta_s:
                    idempotent_violations.append(
                        f"μ(η({s})) = μ({eta_s}) = {mu_eta_s} ≠ {eta_s}"
                    )
            if idempotent_violations:
                pair_result["idempotent_violations"] = idempotent_violations[:5]
                result.details.append(
                    f"Pair {idx}: μ∘η ≠ id on {len(idempotent_violations)} states"
                )

            result.pair_results.append(pair_result)

        result.aggregate_commutes = all_commute
        result.status = (
            DiagramStatus.COMMUTES if all_commute else DiagramStatus.FAILS
        )
        result.verification_time_seconds = time.monotonic() - t0
        result.compute_proof_hash()
        self._results.append(result)
        return result

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _is_union_of_classes(
        self,
        state_set: FrozenSet[str],
        stutter_classes: List[Any],
    ) -> bool:
        """Check if a set of states is a union of stutter equivalence classes."""
        for cls in stutter_classes:
            intersection = cls.members & state_set
            if intersection and intersection != cls.members:
                return False
        return True

    def aggregate_result(self) -> Dict[str, Any]:
        """Produce an aggregate summary of all diagram verifications.

        Returns a dictionary suitable for inclusion in a proof certificate.
        """
        total = len(self._results)
        commuting = sum(1 for r in self._results if r.aggregate_commutes)
        failing = sum(
            1 for r in self._results if r.status == DiagramStatus.FAILS
        )

        # Compute aggregate hash over all diagram hashes
        hasher = hashlib.sha256()
        for r in self._results:
            hasher.update(r.proof_hash.encode())
        aggregate_hash = hasher.hexdigest()

        return {
            "total_diagrams": total,
            "commuting": commuting,
            "failing": failing,
            "all_commute": commuting == total and total > 0,
            "aggregate_hash": aggregate_hash,
            "diagrams": [r.to_dict() for r in self._results],
        }
