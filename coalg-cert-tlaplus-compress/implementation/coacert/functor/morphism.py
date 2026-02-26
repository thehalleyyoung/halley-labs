"""
Coalgebra morphism computation for CoaCert-TLA.

A coalgebra morphism h: (S1, γ1) → (S2, γ2) is a function h: S1 → S2
such that γ2 ∘ h = F(h) ∘ γ1, i.e., the following diagram commutes:

    S1 ---γ1--→ F(S1)
    |             |
    h           F(h)
    |             |
    v             v
    S2 ---γ2--→ F(S2)

For F(X) = P(AP) × P(X)^Act × Fair(X), this means:
  - AP preservation: L(s) = L(h(s))
  - Successor preservation: h(succ(s,a)) = succ(h(s),a) as sets
  - Fairness preservation: fairness membership is preserved
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
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
# Morphism representation
# ---------------------------------------------------------------------------

@dataclass
class CoalgebraMorphism:
    """A coalgebra morphism h: source → target."""

    source_name: str
    target_name: str
    mapping: Dict[str, str]
    is_valid: bool = True
    properties: Dict[str, bool] = field(default_factory=dict)

    def apply(self, state: str) -> str:
        return self.mapping[state]

    def domain(self) -> FrozenSet[str]:
        return frozenset(self.mapping.keys())

    def codomain(self) -> FrozenSet[str]:
        return frozenset(self.mapping.values())

    def image(self) -> FrozenSet[str]:
        return frozenset(self.mapping.values())

    def is_injective(self) -> bool:
        return len(set(self.mapping.values())) == len(self.mapping)

    def is_surjective(self, target_states: FrozenSet[str]) -> bool:
        return self.image() >= target_states

    def kernel(self) -> List[FrozenSet[str]]:
        """Compute the kernel partition: states mapped to the same target."""
        classes: Dict[str, Set[str]] = defaultdict(set)
        for s, t in self.mapping.items():
            classes[t].add(s)
        return [frozenset(cls) for cls in classes.values()]

    def fibers(self) -> Dict[str, FrozenSet[str]]:
        """For each target state, return the set of source states mapping to it."""
        fib: Dict[str, Set[str]] = defaultdict(set)
        for s, t in self.mapping.items():
            fib[t].add(s)
        return {t: frozenset(ss) for t, ss in fib.items()}

    def restrict(self, subset: FrozenSet[str]) -> "CoalgebraMorphism":
        """Restrict the morphism to a subset of the domain."""
        restricted = {s: t for s, t in self.mapping.items() if s in subset}
        return CoalgebraMorphism(
            source_name=self.source_name,
            target_name=self.target_name,
            mapping=restricted,
        )


class MorphismComposition:
    """Compose two coalgebra morphisms."""

    @staticmethod
    def compose(
        f: CoalgebraMorphism, g: CoalgebraMorphism
    ) -> CoalgebraMorphism:
        """Compute g ∘ f: source_f → target_g.

        Requires: target of f = source of g (same state space).
        """
        composed: Dict[str, str] = {}
        for s, fs in f.mapping.items():
            if fs in g.mapping:
                composed[s] = g.mapping[fs]
            else:
                logger.warning(
                    "Composition gap: f(%s)=%s not in domain of g", s, fs
                )

        return CoalgebraMorphism(
            source_name=f.source_name,
            target_name=g.target_name,
            mapping=composed,
            is_valid=f.is_valid and g.is_valid,
        )

    @staticmethod
    def identity(coalgebra_name: str, states: FrozenSet[str]) -> CoalgebraMorphism:
        """Create the identity morphism."""
        return CoalgebraMorphism(
            source_name=coalgebra_name,
            target_name=coalgebra_name,
            mapping={s: s for s in states},
            is_valid=True,
            properties={"injective": True, "surjective": True, "isomorphism": True},
        )


# ---------------------------------------------------------------------------
# Morphism finder
# ---------------------------------------------------------------------------

class MorphismFinder:
    """Find coalgebra morphisms between two coalgebras.

    Uses constraint propagation and backtracking to find
    functions h: S1 → S2 satisfying the morphism conditions.
    """

    def __init__(self, source: Any, target: Any):
        self.source = source
        self.target = target
        self._candidates: Dict[str, Set[str]] = {}
        self._found: Optional[CoalgebraMorphism] = None

    def find_morphism(self) -> Optional[CoalgebraMorphism]:
        """Find a coalgebra morphism from source to target, if one exists.

        Returns None if no morphism exists.
        """
        self._initialize_candidates()

        if not self._propagate_constraints():
            logger.info("No morphism exists: constraint propagation failed")
            return None

        assignment: Dict[str, str] = {}
        if self._backtrack(assignment):
            morphism = CoalgebraMorphism(
                source_name=self.source.name,
                target_name=self.target.name,
                mapping=assignment,
            )
            morphism.properties = self._classify_morphism(morphism)
            self._found = morphism
            return morphism

        logger.info("No morphism exists: backtracking exhausted")
        return None

    def _initialize_candidates(self) -> None:
        """Initialize candidate mappings based on AP labels."""
        target_by_label: Dict[FrozenSet[str], Set[str]] = defaultdict(set)
        for t in self.target.states:
            t_fv = self.target.apply_functor(t)
            target_by_label[t_fv.propositions].add(t)

        for s in self.source.states:
            s_fv = self.source.apply_functor(s)
            candidates = target_by_label.get(s_fv.propositions, set())
            self._candidates[s] = set(candidates)

        logger.debug(
            "Initialized candidates: avg %.1f candidates per state",
            sum(len(c) for c in self._candidates.values())
            / max(len(self._candidates), 1),
        )

    def _propagate_constraints(self) -> bool:
        """Arc consistency propagation (AC-3 style).

        Reduce candidate sets using the successor preservation constraint.
        """
        queue: Deque[str] = deque(self.source.states)
        iterations = 0
        max_iterations = len(self.source.states) * len(self.source.actions) * 10

        while queue and iterations < max_iterations:
            iterations += 1
            s = queue.popleft()

            if not self._candidates[s]:
                return False

            s_fv = self.source.apply_functor(s)
            changed = False

            to_remove: Set[str] = set()
            for t_cand in self._candidates[s]:
                t_fv = self.target.apply_functor(t_cand)

                if not self._is_locally_consistent(s_fv, t_fv):
                    to_remove.add(t_cand)
                    changed = True

            self._candidates[s] -= to_remove

            if not self._candidates[s]:
                return False

            if changed:
                preds = self._source_predecessors(s)
                for p in preds:
                    if p not in queue:
                        queue.append(p)

        return True

    def _is_locally_consistent(
        self, s_fv: Any, t_fv: Any
    ) -> bool:
        """Check local consistency of mapping s → t.

        Verifies that for each action, the successor set constraints
        can potentially be satisfied.
        """
        if s_fv.propositions != t_fv.propositions:
            return False

        for act in s_fv.actions():
            s_succs = s_fv.successor_set(act)
            t_succs = t_fv.successor_set(act)

            for s_next in s_succs:
                if s_next in self._candidates:
                    if not (self._candidates[s_next] & t_succs):
                        return False

        for idx in s_fv.fairness_membership:
            s_mem = s_fv.fairness_membership.get(idx, (False, False))
            t_mem = t_fv.fairness_membership.get(idx, (False, False))
            if s_mem != t_mem:
                return False

        return True

    def _source_predecessors(self, state: str) -> Set[str]:
        """Find predecessors of a state in the source coalgebra."""
        preds: Set[str] = set()
        for s in self.source.states:
            if state in self.source.all_successors(s):
                preds.add(s)
        return preds

    def _backtrack(self, assignment: Dict[str, str]) -> bool:
        """Backtracking search for a valid morphism."""
        if len(assignment) == len(self.source.states):
            return self._verify_assignment(assignment)

        unassigned = [
            s for s in self.source.states if s not in assignment
        ]
        if not unassigned:
            return True

        # choose variable with fewest candidates (MRV heuristic)
        var = min(unassigned, key=lambda s: len(self._candidates[s]))

        for t_cand in sorted(self._candidates[var]):
            assignment[var] = t_cand

            if self._is_assignment_consistent(var, t_cand, assignment):
                if self._backtrack(assignment):
                    return True

            del assignment[var]

        return False

    def _is_assignment_consistent(
        self, state: str, target: str, assignment: Dict[str, str]
    ) -> bool:
        """Check if assigning state → target is consistent with
        existing assignments.
        """
        s_fv = self.source.apply_functor(state)
        t_fv = self.target.apply_functor(target)

        if s_fv.propositions != t_fv.propositions:
            return False

        for act in s_fv.actions():
            s_succs = s_fv.successor_set(act)
            t_succs = t_fv.successor_set(act)

            for s_next in s_succs:
                if s_next in assignment:
                    if assignment[s_next] not in t_succs:
                        return False

        for idx in s_fv.fairness_membership:
            s_mem = s_fv.fairness_membership.get(idx, (False, False))
            t_mem = t_fv.fairness_membership.get(idx, (False, False))
            if s_mem != t_mem:
                return False

        return True

    def _verify_assignment(self, assignment: Dict[str, str]) -> bool:
        """Full verification that a complete assignment is a valid morphism."""
        for s in self.source.states:
            hs = assignment[s]
            s_fv = self.source.apply_functor(s)
            t_fv = self.target.apply_functor(hs)

            if s_fv.propositions != t_fv.propositions:
                return False

            for act in s_fv.actions() | t_fv.actions():
                s_succs = s_fv.successor_set(act)
                mapped_succs = frozenset(assignment.get(x, x) for x in s_succs)
                t_succs = t_fv.successor_set(act)
                if mapped_succs != t_succs:
                    return False

            for idx in set(s_fv.fairness_membership) | set(t_fv.fairness_membership):
                s_mem = s_fv.fairness_membership.get(idx, (False, False))
                t_mem = t_fv.fairness_membership.get(idx, (False, False))
                if s_mem != t_mem:
                    return False

        return True

    def _classify_morphism(
        self, morph: CoalgebraMorphism
    ) -> Dict[str, bool]:
        """Classify the morphism as mono/epi/iso."""
        is_mono = morph.is_injective()
        is_epi = morph.is_surjective(self.target.states)
        return {
            "injective": is_mono,
            "surjective": is_epi,
            "isomorphism": is_mono and is_epi,
            "monomorphism": is_mono,
            "epimorphism": is_epi,
        }

    # -- verification -------------------------------------------------------

    def verify_morphism(self, morphism: CoalgebraMorphism) -> Tuple[bool, List[str]]:
        """Verify that a given morphism is valid."""
        issues: List[str] = []

        for s in self.source.states:
            if s not in morphism.mapping:
                issues.append(f"State {s} not in mapping domain")
                continue

            hs = morphism.mapping[s]
            if hs not in self.target.states:
                issues.append(f"h({s})={hs} not in target states")
                continue

            s_fv = self.source.apply_functor(s)
            t_fv = self.target.apply_functor(hs)

            if s_fv.propositions != t_fv.propositions:
                issues.append(
                    f"AP mismatch at {s}: L({s})={s_fv.propositions} ≠ L(h({s}))={t_fv.propositions}"
                )

            for act in s_fv.actions() | t_fv.actions():
                s_succs = s_fv.successor_set(act)
                mapped = frozenset(morphism.mapping.get(x, x) for x in s_succs)
                t_succs = t_fv.successor_set(act)
                if mapped != t_succs:
                    issues.append(
                        f"Successor mismatch at ({s},{act}): "
                        f"h(succ)={sorted(mapped)} ≠ succ(h)={sorted(t_succs)}"
                    )

            for idx in set(s_fv.fairness_membership) | set(t_fv.fairness_membership):
                s_mem = s_fv.fairness_membership.get(idx, (False, False))
                t_mem = t_fv.fairness_membership.get(idx, (False, False))
                if s_mem != t_mem:
                    issues.append(
                        f"Fairness mismatch at {s}, pair {idx}: {s_mem} ≠ {t_mem}"
                    )

        return len(issues) == 0, issues

    # -- minimal morphism (to quotient) -------------------------------------

    def find_minimal_morphism(
        self, partition: List[FrozenSet[str]]
    ) -> CoalgebraMorphism:
        """Compute the canonical morphism from source to its quotient
        by the given partition.

        This is always the natural projection.
        """
        mapping: Dict[str, str] = {}
        for block in partition:
            rep = min(block)
            for s in block:
                mapping[s] = rep

        morph = CoalgebraMorphism(
            source_name=self.source.name,
            target_name=f"{self.source.name}/~",
            mapping=mapping,
        )
        morph.properties = {
            "surjective": True,
            "epimorphism": True,
            "injective": all(len(b) == 1 for b in partition),
            "isomorphism": all(len(b) == 1 for b in partition),
        }
        return morph

    # -- find all morphisms (for small systems) -----------------------------

    def find_all_morphisms(
        self, limit: int = 100
    ) -> List[CoalgebraMorphism]:
        """Find all morphisms from source to target (up to limit)."""
        self._initialize_candidates()
        if not self._propagate_constraints():
            return []

        results: List[CoalgebraMorphism] = []
        self._enumerate_all({}, results, limit)
        return results

    def _enumerate_all(
        self,
        assignment: Dict[str, str],
        results: List[CoalgebraMorphism],
        limit: int,
    ) -> None:
        if len(results) >= limit:
            return

        if len(assignment) == len(self.source.states):
            if self._verify_assignment(assignment):
                morph = CoalgebraMorphism(
                    source_name=self.source.name,
                    target_name=self.target.name,
                    mapping=dict(assignment),
                )
                morph.properties = self._classify_morphism(morph)
                results.append(morph)
            return

        unassigned = [s for s in self.source.states if s not in assignment]
        if not unassigned:
            return

        var = min(unassigned, key=lambda s: len(self._candidates[s]))

        for t_cand in sorted(self._candidates[var]):
            assignment[var] = t_cand
            if self._is_assignment_consistent(var, t_cand, assignment):
                self._enumerate_all(assignment, results, limit)
            del assignment[var]
