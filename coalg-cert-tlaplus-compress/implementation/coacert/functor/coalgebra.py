"""
F-coalgebra representation for CoaCert-TLA.

An F-coalgebra is a pair (S, γ: S → F(S)) where:
  F(X) = P(AP) × P(X)^Act × Fair(X)
  - P(AP): powerset of atomic propositions labeling each state
  - P(X)^Act: action-labeled nondeterministic successor function
  - Fair(X): Streett/Rabin acceptance pairs (B_i, G_i) ⊆ X × X

This module provides construction, morphism checking, quotient, sub-coalgebra,
and product operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

S = TypeVar("S")
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FairnessConstraint:
    """A single acceptance pair (B, G) where B and G are sets of states.

    In a Streett condition, a run is accepting if for every pair,
    visiting B infinitely often implies visiting G infinitely often.
    """

    b_states: FrozenSet[str]
    g_states: FrozenSet[str]
    index: int = 0

    def contains_b(self, state: str) -> bool:
        return state in self.b_states

    def contains_g(self, state: str) -> bool:
        return state in self.g_states

    def remap(self, mapping: Mapping[str, str]) -> "FairnessConstraint":
        """Remap state names through a function."""
        new_b = frozenset(mapping.get(s, s) for s in self.b_states)
        new_g = frozenset(mapping.get(s, s) for s in self.g_states)
        return FairnessConstraint(b_states=new_b, g_states=new_g, index=self.index)

    def restrict(self, states: FrozenSet[str]) -> "FairnessConstraint":
        """Restrict to a subset of states."""
        return FairnessConstraint(
            b_states=self.b_states & states,
            g_states=self.g_states & states,
            index=self.index,
        )


@dataclass(frozen=True)
class FunctorValue:
    """The image of a state under the structure map γ.

    For F(X) = P(AP) × P(X)^Act × Fair(X):
      - propositions: the atomic propositions labeling the state
      - successors: for each action, the set of successor state names
      - fairness_membership: for each pair index i, whether this state is
        in B_i and/or G_i
    """

    propositions: FrozenSet[str]
    successors: Dict[str, FrozenSet[str]]
    fairness_membership: Dict[int, Tuple[bool, bool]]  # index -> (in_B, in_G)

    def successor_set(self, action: str) -> FrozenSet[str]:
        return self.successors.get(action, frozenset())

    def all_successors(self) -> FrozenSet[str]:
        result: Set[str] = set()
        for targets in self.successors.values():
            result |= targets
        return frozenset(result)

    def actions(self) -> FrozenSet[str]:
        return frozenset(self.successors.keys())

    def remap_successors(self, mapping: Mapping[str, str]) -> "FunctorValue":
        new_succ = {}
        for act, targets in self.successors.items():
            new_succ[act] = frozenset(mapping.get(t, t) for t in targets)
        return FunctorValue(
            propositions=self.propositions,
            successors=new_succ,
            fairness_membership=dict(self.fairness_membership),
        )


@dataclass
class CoalgebraState:
    """A state together with its F-structure."""

    name: str
    propositions: FrozenSet[str] = field(default_factory=frozenset)
    successors: Dict[str, Set[str]] = field(default_factory=dict)
    fairness_membership: Dict[int, Tuple[bool, bool]] = field(default_factory=dict)

    def functor_value(self) -> FunctorValue:
        frozen_succ = {
            act: frozenset(targets) for act, targets in self.successors.items()
        }
        return FunctorValue(
            propositions=self.propositions,
            successors=frozen_succ,
            fairness_membership=dict(self.fairness_membership),
        )

    def add_successor(self, action: str, target: str) -> None:
        self.successors.setdefault(action, set()).add(target)

    def set_fairness(self, index: int, in_b: bool, in_g: bool) -> None:
        self.fairness_membership[index] = (in_b, in_g)


# ---------------------------------------------------------------------------
# Transition graph interface
# ---------------------------------------------------------------------------

@dataclass
class TransitionGraph:
    """A labeled transition system with nondeterministic transitions.

    This is the *input* from which we construct an F-coalgebra.
    """

    states: Set[str] = field(default_factory=set)
    initial_states: Set[str] = field(default_factory=set)
    actions: Set[str] = field(default_factory=set)
    transitions: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    labels: Dict[str, Set[str]] = field(default_factory=dict)

    def add_state(self, name: str, props: Optional[Set[str]] = None) -> None:
        self.states.add(name)
        if props is not None:
            self.labels[name] = props

    def add_transition(self, src: str, action: str, dst: str) -> None:
        self.states.add(src)
        self.states.add(dst)
        self.actions.add(action)
        self.transitions.setdefault(src, {}).setdefault(action, set()).add(dst)

    def get_successors(self, state: str, action: str) -> Set[str]:
        return self.transitions.get(state, {}).get(action, set())

    def get_labels(self, state: str) -> Set[str]:
        return self.labels.get(state, set())

    def all_successors(self, state: str) -> Set[str]:
        result: Set[str] = set()
        for targets in self.transitions.get(state, {}).values():
            result |= targets
        return result

    def reachable_from(self, roots: Set[str]) -> Set[str]:
        visited: Set[str] = set()
        worklist = list(roots)
        while worklist:
            s = worklist.pop()
            if s in visited:
                continue
            visited.add(s)
            for t in self.all_successors(s):
                if t not in visited:
                    worklist.append(t)
        return visited

    def reverse_transitions(self) -> Dict[str, Dict[str, Set[str]]]:
        rev: Dict[str, Dict[str, Set[str]]] = {}
        for src, act_map in self.transitions.items():
            for act, targets in act_map.items():
                for dst in targets:
                    rev.setdefault(dst, {}).setdefault(act, set()).add(src)
        return rev


# ---------------------------------------------------------------------------
# F-Coalgebra
# ---------------------------------------------------------------------------

class FCoalgebra:
    """An F-coalgebra (S, γ: S → F(S)) where F(X) = P(AP) × P(X)^Act × Fair(X).

    The structure map γ sends each state to its FunctorValue, encoding the
    atomic propositions, action-indexed successor sets, and fairness
    membership information.
    """

    def __init__(
        self,
        name: str = "unnamed",
        atomic_propositions: Optional[Set[str]] = None,
        actions: Optional[Set[str]] = None,
    ):
        self.name = name
        self.atomic_propositions: Set[str] = atomic_propositions or set()
        self.actions: Set[str] = actions or set()
        self._states: Dict[str, CoalgebraState] = {}
        self._fairness_constraints: List[FairnessConstraint] = []
        self._initial_states: Set[str] = set()

    # -- state management ---------------------------------------------------

    @property
    def states(self) -> FrozenSet[str]:
        return frozenset(self._states.keys())

    @property
    def state_count(self) -> int:
        return len(self._states)

    @property
    def initial_states(self) -> FrozenSet[str]:
        return frozenset(self._initial_states)

    @property
    def fairness_constraints(self) -> List[FairnessConstraint]:
        return list(self._fairness_constraints)

    def add_state(
        self,
        name: str,
        propositions: Optional[Set[str]] = None,
        is_initial: bool = False,
    ) -> CoalgebraState:
        props = frozenset(propositions) if propositions else frozenset()
        self.atomic_propositions |= set(props)
        cs = CoalgebraState(name=name, propositions=props)
        self._states[name] = cs
        if is_initial:
            self._initial_states.add(name)
        return cs

    def get_state(self, name: str) -> Optional[CoalgebraState]:
        return self._states.get(name)

    def add_transition(self, src: str, action: str, dst: str) -> None:
        if src not in self._states:
            self.add_state(src)
        if dst not in self._states:
            self.add_state(dst)
        self.actions.add(action)
        self._states[src].add_successor(action, dst)

    def add_fairness_constraint(
        self, b_states: Set[str], g_states: Set[str]
    ) -> int:
        idx = len(self._fairness_constraints)
        fc = FairnessConstraint(
            b_states=frozenset(b_states),
            g_states=frozenset(g_states),
            index=idx,
        )
        self._fairness_constraints.append(fc)
        for s in self._states.values():
            s.set_fairness(idx, s.name in b_states, s.name in g_states)
        return idx

    def set_initial(self, name: str) -> None:
        if name in self._states:
            self._initial_states.add(name)

    # -- structure map γ ----------------------------------------------------

    def apply_functor(self, state: str) -> FunctorValue:
        """Compute γ(state), the image of the structure map."""
        cs = self._states.get(state)
        if cs is None:
            raise KeyError(f"State '{state}' not in coalgebra")
        return cs.functor_value()

    def structure_map(self) -> Dict[str, FunctorValue]:
        """Return the full structure map γ as a dictionary."""
        return {name: cs.functor_value() for name, cs in self._states.items()}

    # -- construction from transition graph ---------------------------------

    @classmethod
    def from_transition_graph(
        cls,
        graph: TransitionGraph,
        fairness: Optional[List[Tuple[Set[str], Set[str]]]] = None,
        name: str = "from_graph",
    ) -> "FCoalgebra":
        """Construct an F-coalgebra from a TransitionGraph and optional fairness pairs."""
        coalg = cls(
            name=name,
            atomic_propositions=set(),
            actions=set(graph.actions),
        )
        for s in graph.states:
            props = graph.get_labels(s)
            is_init = s in graph.initial_states
            coalg.add_state(s, propositions=props, is_initial=is_init)

        for src, act_map in graph.transitions.items():
            for act, targets in act_map.items():
                for dst in targets:
                    coalg.add_transition(src, act, dst)

        if fairness:
            for b_set, g_set in fairness:
                coalg.add_fairness_constraint(b_set, g_set)

        logger.info(
            "Constructed coalgebra '%s' with %d states, %d actions, %d fairness pairs",
            name,
            coalg.state_count,
            len(coalg.actions),
            len(coalg._fairness_constraints),
        )
        return coalg

    # -- morphism checking --------------------------------------------------

    def verify_morphism_to(
        self,
        target: "FCoalgebra",
        h: Mapping[str, str],
    ) -> "MorphismCheckResult":
        """Check whether h: self → target is a coalgebra morphism.

        A morphism must satisfy for every state s in self:
          1) L(s) = L(h(s))                              (AP preservation)
          2) h(succ(s,a)) = succ(h(s),a) for all a       (successor preservation)
          3) fairness membership is preserved              (fairness preservation)
        """
        checker = CoalgebraMorphismCheck(source=self, target=target, mapping=h)
        return checker.check()

    # -- successor queries --------------------------------------------------

    def successors(self, state: str, action: str) -> FrozenSet[str]:
        cs = self._states.get(state)
        if cs is None:
            return frozenset()
        return frozenset(cs.successors.get(action, set()))

    def all_actions_from(self, state: str) -> FrozenSet[str]:
        cs = self._states.get(state)
        if cs is None:
            return frozenset()
        return frozenset(cs.successors.keys())

    def all_successors(self, state: str) -> FrozenSet[str]:
        cs = self._states.get(state)
        if cs is None:
            return frozenset()
        result: Set[str] = set()
        for targets in cs.successors.values():
            result |= targets
        return frozenset(result)

    def predecessors(self, state: str) -> Dict[str, Set[str]]:
        """Compute predecessor map {action: {predecessor states}}."""
        preds: Dict[str, Set[str]] = {}
        for s_name, cs in self._states.items():
            for act, targets in cs.successors.items():
                if state in targets:
                    preds.setdefault(act, set()).add(s_name)
        return preds

    # -- reachability -------------------------------------------------------

    def reachable_states(self, roots: Optional[Set[str]] = None) -> FrozenSet[str]:
        if roots is None:
            roots = set(self._initial_states)
        visited: Set[str] = set()
        worklist = list(roots)
        while worklist:
            s = worklist.pop()
            if s in visited:
                continue
            visited.add(s)
            for t in self.all_successors(s):
                if t not in visited:
                    worklist.append(t)
        return frozenset(visited)

    def trim(self) -> "FCoalgebra":
        """Return a new coalgebra restricted to reachable states."""
        reachable = self.reachable_states()
        return self.restrict_to(reachable)

    def restrict_to(self, state_set: FrozenSet[str]) -> "FCoalgebra":
        """Return a sub-coalgebra restricted to a set of states."""
        sub = FCoalgebra(
            name=f"{self.name}_restricted",
            atomic_propositions=set(self.atomic_propositions),
            actions=set(self.actions),
        )
        for s in state_set:
            cs = self._states.get(s)
            if cs is None:
                continue
            sub.add_state(s, propositions=set(cs.propositions), is_initial=(s in self._initial_states))
            for act, targets in cs.successors.items():
                for t in targets:
                    if t in state_set:
                        sub.add_transition(s, act, t)

        for fc in self._fairness_constraints:
            restricted = fc.restrict(state_set)
            sub.add_fairness_constraint(set(restricted.b_states), set(restricted.g_states))

        return sub

    # -- equivalence relation & quotient ------------------------------------

    def quotient(
        self,
        partition: List[FrozenSet[str]],
        representative: Optional[Callable[[FrozenSet[str]], str]] = None,
    ) -> Tuple["FCoalgebra", Dict[str, str]]:
        """Construct the quotient coalgebra given an equivalence relation
        represented as a partition (list of equivalence classes).

        Returns (quotient_coalgebra, projection_map).
        """
        return QuotientCoalgebra.build(self, partition, representative)

    # -- sub-coalgebra extraction -------------------------------------------

    def sub_coalgebra(self, roots: Set[str]) -> "FCoalgebra":
        """Extract the sub-coalgebra generated by a set of root states."""
        return SubCoalgebra.extract(self, roots)

    # -- product construction -----------------------------------------------

    @staticmethod
    def product(c1: "FCoalgebra", c2: "FCoalgebra") -> Tuple["FCoalgebra", Dict[str, Tuple[str, str]]]:
        """Construct the product coalgebra c1 × c2."""
        return ProductCoalgebra.build(c1, c2)

    # -- utility ------------------------------------------------------------

    def to_transition_graph(self) -> TransitionGraph:
        """Convert back to a TransitionGraph."""
        g = TransitionGraph()
        g.states = set(self._states.keys())
        g.initial_states = set(self._initial_states)
        g.actions = set(self.actions)
        for s, cs in self._states.items():
            g.labels[s] = set(cs.propositions)
            for act, targets in cs.successors.items():
                for t in targets:
                    g.add_transition(s, act, t)
        return g

    def __repr__(self) -> str:
        return (
            f"FCoalgebra(name={self.name!r}, states={self.state_count}, "
            f"actions={len(self.actions)}, fairness_pairs={len(self._fairness_constraints)})"
        )

    def state_signature(self, state: str) -> Tuple:
        """Compute a hashable signature for a state (for partition refinement)."""
        fv = self.apply_functor(state)
        succ_tuple = tuple(sorted(
            (act, tuple(sorted(targets))) for act, targets in fv.successors.items()
        ))
        fair_tuple = tuple(sorted(fv.fairness_membership.items()))
        return (fv.propositions, succ_tuple, fair_tuple)

    def is_deterministic(self) -> bool:
        """Check if the coalgebra is deterministic (at most one successor per action)."""
        for cs in self._states.values():
            for targets in cs.successors.values():
                if len(targets) > 1:
                    return False
        return True

    def is_total(self) -> bool:
        """Check if every state has at least one successor for every action."""
        for cs in self._states.values():
            for act in self.actions:
                if not cs.successors.get(act):
                    return False
        return True

    def deadlock_states(self) -> FrozenSet[str]:
        """Return states with no outgoing transitions."""
        dead: Set[str] = set()
        for name, cs in self._states.items():
            has_succ = any(bool(targets) for targets in cs.successors.values())
            if not has_succ:
                dead.add(name)
        return frozenset(dead)

    def strongly_connected_components(self) -> List[FrozenSet[str]]:
        """Compute SCCs using Tarjan's algorithm."""
        index_counter = [0]
        stack: List[str] = []
        on_stack: Set[str] = set()
        index_map: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        sccs: List[FrozenSet[str]] = []

        def strongconnect(v: str) -> None:
            index_map[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in self.all_successors(v):
                if w not in index_map:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index_map[w])

            if lowlink[v] == index_map[v]:
                component: Set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.add(w)
                    if w == v:
                        break
                sccs.append(frozenset(component))

        for v in sorted(self._states.keys()):
            if v not in index_map:
                strongconnect(v)

        return sccs

    def accepting_sccs(self, condition: str = "streett") -> List[FrozenSet[str]]:
        """Find SCCs that satisfy the acceptance condition.

        For Streett: an SCC is accepting if for every pair (B_i, G_i),
        if the SCC intersects B_i then it also intersects G_i.
        """
        sccs = self.strongly_connected_components()
        accepting = []
        for scc in sccs:
            if len(scc) == 1:
                s = next(iter(scc))
                if s not in self.all_successors(s):
                    continue  # trivial SCC with no self-loop

            is_accepting = True
            for fc in self._fairness_constraints:
                if condition == "streett":
                    intersects_b = bool(scc & fc.b_states)
                    intersects_g = bool(scc & fc.g_states)
                    if intersects_b and not intersects_g:
                        is_accepting = False
                        break
                elif condition == "rabin":
                    intersects_b = bool(scc & fc.b_states)
                    intersects_g = bool(scc & fc.g_states)
                    if not intersects_b and intersects_g:
                        is_accepting = True
                        break
                    is_accepting = False

            if is_accepting:
                accepting.append(scc)

        return accepting


# ---------------------------------------------------------------------------
# Morphism check
# ---------------------------------------------------------------------------

@dataclass
class MorphismCheckResult:
    """Result of checking whether a function is a coalgebra morphism."""

    is_morphism: bool
    violations: List[str] = field(default_factory=list)
    ap_violations: List[Tuple[str, FrozenSet[str], FrozenSet[str]]] = field(
        default_factory=list
    )
    successor_violations: List[Tuple[str, str, FrozenSet[str], FrozenSet[str]]] = field(
        default_factory=list
    )
    fairness_violations: List[Tuple[str, int]] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_morphism:
            return "Valid coalgebra morphism"
        parts = [f"Not a morphism: {len(self.violations)} violation(s)"]
        if self.ap_violations:
            parts.append(f"  AP violations: {len(self.ap_violations)}")
        if self.successor_violations:
            parts.append(f"  Successor violations: {len(self.successor_violations)}")
        if self.fairness_violations:
            parts.append(f"  Fairness violations: {len(self.fairness_violations)}")
        return "\n".join(parts)


class CoalgebraMorphismCheck:
    """Verify that a mapping h: S1 → S2 is a coalgebra morphism."""

    def __init__(
        self,
        source: FCoalgebra,
        target: FCoalgebra,
        mapping: Mapping[str, str],
    ):
        self.source = source
        self.target = target
        self.mapping = mapping

    def check(self) -> MorphismCheckResult:
        result = MorphismCheckResult(is_morphism=True)

        for s in self.source.states:
            if s not in self.mapping:
                result.is_morphism = False
                result.violations.append(f"State '{s}' not in mapping domain")
                continue

            hs = self.mapping[s]
            if hs not in self.target.states:
                result.is_morphism = False
                result.violations.append(
                    f"h({s})={hs} not in target states"
                )
                continue

            self._check_ap_preservation(s, hs, result)
            self._check_successor_preservation(s, hs, result)
            self._check_fairness_preservation(s, hs, result)

        return result

    def _check_ap_preservation(
        self, s: str, hs: str, result: MorphismCheckResult
    ) -> None:
        src_fv = self.source.apply_functor(s)
        tgt_fv = self.target.apply_functor(hs)
        if src_fv.propositions != tgt_fv.propositions:
            result.is_morphism = False
            result.ap_violations.append((s, src_fv.propositions, tgt_fv.propositions))
            result.violations.append(
                f"AP mismatch at {s}: {src_fv.propositions} ≠ {tgt_fv.propositions}"
            )

    def _check_successor_preservation(
        self, s: str, hs: str, result: MorphismCheckResult
    ) -> None:
        src_fv = self.source.apply_functor(s)
        tgt_fv = self.target.apply_functor(hs)

        all_actions = src_fv.actions() | tgt_fv.actions()
        for act in all_actions:
            src_succs = src_fv.successor_set(act)
            mapped_succs = frozenset(self.mapping.get(t, t) for t in src_succs)
            tgt_succs = tgt_fv.successor_set(act)

            if mapped_succs != tgt_succs:
                result.is_morphism = False
                result.successor_violations.append(
                    (s, act, mapped_succs, tgt_succs)
                )
                result.violations.append(
                    f"Successor mismatch at ({s}, {act}): "
                    f"h(succ({s},{act}))={mapped_succs} ≠ succ(h({s}),{act})={tgt_succs}"
                )

    def _check_fairness_preservation(
        self, s: str, hs: str, result: MorphismCheckResult
    ) -> None:
        src_fv = self.source.apply_functor(s)
        tgt_fv = self.target.apply_functor(hs)

        all_indices = set(src_fv.fairness_membership.keys()) | set(
            tgt_fv.fairness_membership.keys()
        )
        for idx in all_indices:
            src_membership = src_fv.fairness_membership.get(idx, (False, False))
            tgt_membership = tgt_fv.fairness_membership.get(idx, (False, False))
            if src_membership != tgt_membership:
                result.is_morphism = False
                result.fairness_violations.append((s, idx))
                result.violations.append(
                    f"Fairness mismatch at {s}, pair {idx}: "
                    f"{src_membership} ≠ {tgt_membership}"
                )


# ---------------------------------------------------------------------------
# Quotient coalgebra
# ---------------------------------------------------------------------------

class QuotientCoalgebra:
    """Construct the quotient coalgebra S/~ from a partition."""

    @staticmethod
    def build(
        coalg: FCoalgebra,
        partition: List[FrozenSet[str]],
        representative: Optional[Callable[[FrozenSet[str]], str]] = None,
    ) -> Tuple[FCoalgebra, Dict[str, str]]:
        if representative is None:
            representative = lambda block: min(block)

        state_to_block: Dict[str, int] = {}
        for i, block in enumerate(partition):
            for s in block:
                state_to_block[s] = i

        covered = set()
        for block in partition:
            covered |= block
        if covered != coalg.states:
            missing = coalg.states - covered
            extra = covered - coalg.states
            if missing:
                raise ValueError(f"Partition does not cover states: {missing}")
            if extra:
                raise ValueError(f"Partition contains unknown states: {extra}")

        reps: Dict[int, str] = {}
        for i, block in enumerate(partition):
            reps[i] = representative(block)

        projection: Dict[str, str] = {}
        for s in coalg.states:
            projection[s] = reps[state_to_block[s]]

        quot = FCoalgebra(
            name=f"{coalg.name}_quotient",
            atomic_propositions=set(coalg.atomic_propositions),
            actions=set(coalg.actions),
        )

        for i, block in enumerate(partition):
            rep = reps[i]
            rep_state = coalg.get_state(rep)
            if rep_state is None:
                continue
            is_init = any(s in coalg.initial_states for s in block)
            quot.add_state(rep, propositions=set(rep_state.propositions), is_initial=is_init)

        for i, block in enumerate(partition):
            rep = reps[i]
            for s in block:
                cs = coalg.get_state(s)
                if cs is None:
                    continue
                for act, targets in cs.successors.items():
                    for t in targets:
                        t_rep = projection[t]
                        quot.add_transition(rep, act, t_rep)

        for fc in coalg.fairness_constraints:
            new_b = set()
            new_g = set()
            for i, block in enumerate(partition):
                rep = reps[i]
                if block & fc.b_states:
                    new_b.add(rep)
                if block & fc.g_states:
                    new_g.add(rep)
            quot.add_fairness_constraint(new_b, new_g)

        logger.info(
            "Built quotient coalgebra: %d states → %d classes",
            coalg.state_count,
            len(partition),
        )
        return quot, projection

    @staticmethod
    def verify_quotient_morphism(
        coalg: FCoalgebra,
        quotient: FCoalgebra,
        projection: Dict[str, str],
    ) -> MorphismCheckResult:
        """Verify the natural projection is a valid morphism."""
        return coalg.verify_morphism_to(quotient, projection)


# ---------------------------------------------------------------------------
# Sub-coalgebra
# ---------------------------------------------------------------------------

class SubCoalgebra:
    """Extract a sub-coalgebra closed under successors."""

    @staticmethod
    def extract(coalg: FCoalgebra, roots: Set[str]) -> FCoalgebra:
        reachable = coalg.reachable_states(roots)
        return coalg.restrict_to(reachable)

    @staticmethod
    def is_sub_coalgebra(candidate: FCoalgebra, parent: FCoalgebra) -> bool:
        """Check if candidate is a sub-coalgebra of parent.

        Requires: states of candidate ⊆ states of parent and the
        candidate is closed under the structure map of parent.
        """
        if not candidate.states <= parent.states:
            return False

        for s in candidate.states:
            c_fv = candidate.apply_functor(s)
            p_fv = parent.apply_functor(s)

            if c_fv.propositions != p_fv.propositions:
                return False

            for act in c_fv.actions():
                c_succs = c_fv.successor_set(act)
                p_succs = p_fv.successor_set(act)
                if not c_succs <= p_succs:
                    return False
                if not c_succs <= candidate.states:
                    return False

        return True

    @staticmethod
    def maximal_sub_coalgebra(
        coalg: FCoalgebra, predicate: Callable[[str], bool]
    ) -> FCoalgebra:
        """Compute the largest sub-coalgebra whose states satisfy a predicate.

        Iteratively remove states whose successors leave the candidate set.
        """
        candidates = {s for s in coalg.states if predicate(s)}

        changed = True
        while changed:
            changed = False
            to_remove: Set[str] = set()
            for s in candidates:
                for t in coalg.all_successors(s):
                    if t not in candidates:
                        to_remove.add(s)
                        break
            if to_remove:
                candidates -= to_remove
                changed = True

        return coalg.restrict_to(frozenset(candidates))


# ---------------------------------------------------------------------------
# Product coalgebra
# ---------------------------------------------------------------------------

class ProductCoalgebra:
    """Construct the product coalgebra C1 × C2."""

    @staticmethod
    def build(
        c1: FCoalgebra, c2: FCoalgebra
    ) -> Tuple[FCoalgebra, Dict[str, Tuple[str, str]]]:
        common_actions = c1.actions & c2.actions
        if not common_actions:
            common_actions = c1.actions | c2.actions

        prod = FCoalgebra(
            name=f"{c1.name}_x_{c2.name}",
            atomic_propositions=c1.atomic_propositions | c2.atomic_propositions,
            actions=common_actions,
        )

        pair_map: Dict[str, Tuple[str, str]] = {}

        for s1 in c1.states:
            cs1 = c1.get_state(s1)
            if cs1 is None:
                continue
            for s2 in c2.states:
                cs2 = c2.get_state(s2)
                if cs2 is None:
                    continue

                pair_name = f"({s1},{s2})"
                combined_props = set(cs1.propositions) | set(cs2.propositions)
                is_init = s1 in c1.initial_states and s2 in c2.initial_states
                prod.add_state(pair_name, propositions=combined_props, is_initial=is_init)
                pair_map[pair_name] = (s1, s2)

        for pair_name, (s1, s2) in pair_map.items():
            for act in common_actions:
                succs1 = c1.successors(s1, act)
                succs2 = c2.successors(s2, act)
                for t1 in succs1:
                    for t2 in succs2:
                        target_name = f"({t1},{t2})"
                        if target_name in pair_map:
                            prod.add_transition(pair_name, act, target_name)

        for i, fc1 in enumerate(c1.fairness_constraints):
            prod_b: Set[str] = set()
            prod_g: Set[str] = set()
            for pair_name, (s1, s2) in pair_map.items():
                if s1 in fc1.b_states:
                    prod_b.add(pair_name)
                if s1 in fc1.g_states:
                    prod_g.add(pair_name)
            prod.add_fairness_constraint(prod_b, prod_g)

        for j, fc2 in enumerate(c2.fairness_constraints):
            prod_b = set()
            prod_g = set()
            for pair_name, (s1, s2) in pair_map.items():
                if s2 in fc2.b_states:
                    prod_b.add(pair_name)
                if s2 in fc2.g_states:
                    prod_g.add(pair_name)
            prod.add_fairness_constraint(prod_b, prod_g)

        logger.info(
            "Built product coalgebra: %d × %d = %d states",
            c1.state_count,
            c2.state_count,
            prod.state_count,
        )
        return prod, pair_map

    @staticmethod
    def projection_left(
        prod: FCoalgebra, pair_map: Dict[str, Tuple[str, str]], c1: FCoalgebra
    ) -> Dict[str, str]:
        return {p: s1 for p, (s1, _) in pair_map.items()}

    @staticmethod
    def projection_right(
        prod: FCoalgebra, pair_map: Dict[str, Tuple[str, str]], c2: FCoalgebra
    ) -> Dict[str, str]:
        return {p: s2 for p, (_, s2) in pair_map.items()}
