"""
CTL* model checking on coalgebraic quotient systems.

Implements the standard CTL labeling algorithm (EX, EF, EG, EU + A duality)
and a CTL* extension via automata-theoretic (tree-automaton) reduction.
Produces the set of states satisfying a formula and generates counter-
examples for violated properties.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
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

from .temporal_logic import (
    And,
    Atomic,
    ExistsPath,
    FalseFormula,
    Finally,
    ForallPath,
    FormulaVisitor,
    Globally,
    Iff,
    Implies,
    Next,
    Not,
    Or,
    Release,
    TemporalFormula,
    TrueFormula,
    Until,
    WeakUntil,
    atomic_propositions,
    is_ctl,
    is_ltl,
    simplify,
    to_nnf,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Kripke structure adapter
# ============================================================================

@dataclass
class KripkeAdapter:
    """Thin adapter that wraps an FCoalgebra into a Kripke-like interface
    expected by the model-checking algorithms.

    Fields mirror the standard (S, S0, R, L) Kripke structure.
    """

    states: FrozenSet[str]
    initial_states: FrozenSet[str]
    # action → { src → {dst} }
    transitions: Dict[str, Dict[str, Set[str]]]
    labels: Dict[str, FrozenSet[str]]  # state → set of APs

    @classmethod
    def from_coalgebra(cls, coalg: object) -> "KripkeAdapter":
        """Build from an FCoalgebra instance."""
        states = coalg.states  # type: ignore[attr-defined]
        initial = coalg.initial_states  # type: ignore[attr-defined]
        trans: Dict[str, Dict[str, Set[str]]] = {}
        labels: Dict[str, FrozenSet[str]] = {}

        for s in states:
            cs = coalg.get_state(s)  # type: ignore[attr-defined]
            if cs is None:
                continue
            fv = cs.functor_value()
            labels[s] = fv.propositions
            for act in fv.actions():
                act_map = trans.setdefault(act, {})
                act_map.setdefault(s, set()).update(fv.successor_set(act))

        return cls(
            states=states,
            initial_states=initial,
            transitions=trans,
            labels=labels,
        )

    # -- Convenience accessors -----------------------------------------------

    def successors(self, state: str) -> Set[str]:
        """All successor states across all actions."""
        result: Set[str] = set()
        for act_map in self.transitions.values():
            result.update(act_map.get(state, set()))
        return result

    def predecessors(self, state: str) -> Set[str]:
        """All predecessor states (reverse edges) across all actions."""
        result: Set[str] = set()
        for act_map in self.transitions.values():
            for src, dsts in act_map.items():
                if state in dsts:
                    result.add(src)
        return result

    def has_proposition(self, state: str, prop: str) -> bool:
        return prop in self.labels.get(state, frozenset())

    def reverse_graph(self) -> Dict[str, Set[str]]:
        """Build reverse adjacency map (ignoring actions)."""
        rev: Dict[str, Set[str]] = {s: set() for s in self.states}
        for act_map in self.transitions.values():
            for src, dsts in act_map.items():
                for d in dsts:
                    rev.setdefault(d, set()).add(src)
        return rev


# ============================================================================
# CTL model checking result
# ============================================================================

@dataclass
class CTLCheckResult:
    """Result of CTL / CTL* model checking."""

    formula: TemporalFormula
    satisfying_states: FrozenSet[str]
    initial_states: FrozenSet[str]
    holds: bool  # True if all initial states satisfy the formula
    counterexample: Optional[List[str]] = None  # trace for violated property

    @property
    def violating_initial_states(self) -> FrozenSet[str]:
        return self.initial_states - self.satisfying_states

    def summary(self) -> str:
        status = "HOLDS" if self.holds else "VIOLATED"
        n_sat = len(self.satisfying_states)
        total = len(self.initial_states)
        msg = f"[{status}] {self.formula}  ({n_sat} satisfying states)"
        if not self.holds and self.counterexample:
            path_str = " → ".join(self.counterexample)
            msg += f"\n  counterexample: {path_str}"
        return msg


# ============================================================================
# CTL labeling algorithm
# ============================================================================

class CTLLabeler:
    """Classic CTL model-checking via state labeling.

    For each sub-formula, compute the set of states satisfying it bottom-up.
    """

    def __init__(self, kripke: KripkeAdapter) -> None:
        self._k = kripke
        self._cache: Dict[TemporalFormula, FrozenSet[str]] = {}
        self._rev: Optional[Dict[str, Set[str]]] = None

    def _reverse(self) -> Dict[str, Set[str]]:
        if self._rev is None:
            self._rev = self._k.reverse_graph()
        return self._rev

    def label(self, formula: TemporalFormula) -> FrozenSet[str]:
        """Return the set of states satisfying *formula*."""
        if formula in self._cache:
            return self._cache[formula]
        result = self._label_impl(formula)
        self._cache[formula] = result
        return result

    def _label_impl(self, f: TemporalFormula) -> FrozenSet[str]:
        if isinstance(f, TrueFormula):
            return self._k.states

        if isinstance(f, FalseFormula):
            return frozenset()

        if isinstance(f, Atomic):
            return frozenset(
                s for s in self._k.states
                if self._k.has_proposition(s, f.name)
            )

        if isinstance(f, Not):
            inner = self.label(f.child)
            return self._k.states - inner

        if isinstance(f, And):
            return self.label(f.left) & self.label(f.right)

        if isinstance(f, Or):
            return self.label(f.left) | self.label(f.right)

        if isinstance(f, Implies):
            return (self._k.states - self.label(f.left)) | self.label(f.right)

        if isinstance(f, Iff):
            l_set = self.label(f.left)
            r_set = self.label(f.right)
            return (l_set & r_set) | ((self._k.states - l_set) & (self._k.states - r_set))

        if isinstance(f, ExistsPath):
            return self._label_exists_path(f.path_formula)

        if isinstance(f, ForallPath):
            # A ψ ≡ ¬ E ¬ψ
            neg_inner = f.path_formula.negate()
            e_neg = self._label_exists_path(neg_inner)
            return self._k.states - e_neg

        logger.warning("CTL labeler: unhandled formula type %s", type(f).__name__)
        return frozenset()

    # -- Existential path formulas ------------------------------------------

    def _label_exists_path(self, pf: TemporalFormula) -> FrozenSet[str]:
        """Handle E(path_formula) for CTL path formulas."""
        if isinstance(pf, Next):
            return self._label_ex(pf.child)
        if isinstance(pf, Finally):
            return self._label_ef(pf.child)
        if isinstance(pf, Globally):
            return self._label_eg(pf.child)
        if isinstance(pf, Until):
            return self._label_eu(pf.left, pf.right)
        if isinstance(pf, Release):
            # E(φ R ψ) ≡ ¬A(¬φ U ¬ψ)
            # = ¬(¬E(¬(¬φ U ¬ψ))) → use direct: E(φ R ψ) = E(ψ W (φ ∧ ψ))
            return self._label_e_release(pf.left, pf.right)
        if isinstance(pf, WeakUntil):
            # E(φ W ψ) = E(φ U ψ) ∪ EG(φ)
            eu = self._label_eu(pf.left, pf.right)
            eg = self._label_eg(pf.left)
            return eu | eg
        # If the path formula is actually a state formula, just label it
        if pf.is_state_formula():
            return self.label(pf)
        logger.warning("CTL labeler: unhandled path formula %s", type(pf).__name__)
        return frozenset()

    def _label_ex(self, phi: TemporalFormula) -> FrozenSet[str]:
        """EX φ: states with at least one successor satisfying φ."""
        phi_states = self.label(phi)
        rev = self._reverse()
        result: Set[str] = set()
        for s in phi_states:
            result.update(rev.get(s, set()))
        return frozenset(result)

    def _label_ef(self, phi: TemporalFormula) -> FrozenSet[str]:
        """EF φ: backward reachability from φ-states."""
        phi_states = self.label(phi)
        rev = self._reverse()
        visited: Set[str] = set(phi_states)
        queue: deque[str] = deque(phi_states)
        while queue:
            s = queue.popleft()
            for pred in rev.get(s, set()):
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)
        return frozenset(visited)

    def _label_eg(self, phi: TemporalFormula) -> FrozenSet[str]:
        """EG φ: SCC-based algorithm.

        1. Restrict to φ-subgraph.
        2. Find non-trivial SCCs in the subgraph.
        3. Backward-reachable states from those SCCs within the subgraph.
        """
        phi_states = self.label(phi)
        if not phi_states:
            return frozenset()

        # Build adjacency restricted to phi_states
        adj: Dict[str, Set[str]] = {s: set() for s in phi_states}
        for s in phi_states:
            for succ in self._k.successors(s):
                if succ in phi_states:
                    adj[s].add(succ)

        # Tarjan SCC on restricted graph
        sccs = self._tarjan_scc(phi_states, adj)

        # Collect states in non-trivial SCCs
        scc_states: Set[str] = set()
        for scc in sccs:
            if len(scc) > 1:
                scc_states.update(scc)
            elif len(scc) == 1:
                s = next(iter(scc))
                if s in adj.get(s, set()):
                    scc_states.add(s)

        # Backward reachability within phi-subgraph from scc_states
        rev_adj: Dict[str, Set[str]] = {s: set() for s in phi_states}
        for s, succs in adj.items():
            for t in succs:
                rev_adj[t].add(s)

        visited: Set[str] = set(scc_states)
        queue: deque[str] = deque(scc_states)
        while queue:
            s = queue.popleft()
            for pred in rev_adj.get(s, set()):
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)

        return frozenset(visited)

    def _label_eu(self, phi: TemporalFormula, psi: TemporalFormula) -> FrozenSet[str]:
        """E[φ U ψ]: backward reachability from ψ-states through φ-states."""
        phi_states = self.label(phi)
        psi_states = self.label(psi)
        rev = self._reverse()

        visited: Set[str] = set(psi_states)
        queue: deque[str] = deque(psi_states)
        while queue:
            s = queue.popleft()
            for pred in rev.get(s, set()):
                if pred not in visited and pred in phi_states:
                    visited.add(pred)
                    queue.append(pred)
        return frozenset(visited)

    def _label_e_release(self, phi: TemporalFormula, psi: TemporalFormula) -> FrozenSet[str]:
        """E[φ R ψ]: states from which there exists a path where ψ holds
        until (and including when) φ ∧ ψ holds, or ψ holds globally.

        E[φ R ψ] = E[ψ W (φ ∧ ψ)] = E[ψ U (φ ∧ ψ)] ∪ EG(ψ)
        """
        phi_and_psi = And(phi, psi)
        eu = self._label_eu(psi, phi_and_psi)
        eg = self._label_eg(psi)
        return eu | eg

    # -- Tarjan SCC on a restricted graph -----------------------------------

    @staticmethod
    def _tarjan_scc(
        nodes: FrozenSet[str],
        adj: Dict[str, Set[str]],
    ) -> List[FrozenSet[str]]:
        """Tarjan's algorithm on an explicit adjacency map."""
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

            for w in adj.get(v, set()):
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

        for v in sorted(nodes):
            if v not in index_map:
                strongconnect(v)

        return sccs


# ============================================================================
# Counterexample generation
# ============================================================================

class CounterexampleGenerator:
    """Generate witness/counterexample traces for CTL formulas."""

    def __init__(self, kripke: KripkeAdapter) -> None:
        self._k = kripke

    def shortest_path(self, source: str, targets: FrozenSet[str]) -> Optional[List[str]]:
        """BFS shortest path from *source* to any state in *targets*."""
        if source in targets:
            return [source]
        visited: Set[str] = {source}
        parent: Dict[str, str] = {}
        queue: deque[str] = deque([source])
        while queue:
            s = queue.popleft()
            for t in self._k.successors(s):
                if t not in visited:
                    visited.add(t)
                    parent[t] = s
                    if t in targets:
                        path = [t]
                        cur = t
                        while cur in parent:
                            cur = parent[cur]
                            path.append(cur)
                        path.reverse()
                        return path
                    queue.append(t)
        return None

    def ag_counterexample(
        self,
        labeler: CTLLabeler,
        phi: TemporalFormula,
    ) -> Optional[List[str]]:
        """Counterexample for AG φ: shortest path from an initial state
        to a state where ¬φ holds."""
        phi_states = labeler.label(phi)
        bad_states = self._k.states - phi_states
        if not bad_states:
            return None
        for init in sorted(self._k.initial_states):
            path = self.shortest_path(init, bad_states)
            if path is not None:
                return path
        return None

    def ef_witness(
        self,
        labeler: CTLLabeler,
        phi: TemporalFormula,
    ) -> Optional[List[str]]:
        """Witness for EF φ: path from an initial state to a φ-state."""
        phi_states = labeler.label(phi)
        if not phi_states:
            return None
        for init in sorted(self._k.initial_states):
            path = self.shortest_path(init, phi_states)
            if path is not None:
                return path
        return None

    def eu_witness(
        self,
        labeler: CTLLabeler,
        phi: TemporalFormula,
        psi: TemporalFormula,
    ) -> Optional[List[str]]:
        """Witness for E[φ U ψ]: path where φ holds until ψ."""
        psi_states = labeler.label(psi)
        phi_states = labeler.label(phi)
        for init in sorted(self._k.initial_states):
            path = self._eu_path(init, phi_states, psi_states)
            if path is not None:
                return path
        return None

    def _eu_path(
        self,
        source: str,
        phi_set: FrozenSet[str],
        psi_set: FrozenSet[str],
    ) -> Optional[List[str]]:
        """BFS for E[φ U ψ] witness from *source*."""
        if source in psi_set:
            return [source]
        if source not in phi_set:
            return None
        visited: Set[str] = {source}
        parent: Dict[str, str] = {}
        queue: deque[str] = deque([source])
        while queue:
            s = queue.popleft()
            for t in self._k.successors(s):
                if t in visited:
                    continue
                visited.add(t)
                parent[t] = s
                if t in psi_set:
                    path = [t]
                    cur = t
                    while cur in parent:
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                if t in phi_set:
                    queue.append(t)
        return None

    def eg_lasso(
        self,
        labeler: CTLLabeler,
        phi: TemporalFormula,
    ) -> Optional[Tuple[List[str], List[str]]]:
        """Lasso witness for EG φ: (stem, cycle) where φ holds everywhere.

        Returns (stem, cycle) where stem leads from an initial state to
        the first state of the cycle, and cycle is a sequence of states
        forming a loop where φ holds in all states.
        """
        eg_states = labeler.label(ExistsPath(Globally(phi)))
        if not eg_states:
            return None

        # Find path from initial to an eg_state
        for init in sorted(self._k.initial_states):
            path = self.shortest_path(init, eg_states)
            if path is None:
                continue
            # From the last state of path, find a cycle within eg_states
            cycle = self._find_cycle_in(path[-1], eg_states)
            if cycle is not None:
                return (path, cycle)
        return None

    def _find_cycle_in(self, start: str, allowed: FrozenSet[str]) -> Optional[List[str]]:
        """DFS to find a cycle from *start* staying within *allowed*."""
        visited: Set[str] = set()
        stack: List[str] = [start]
        parent: Dict[str, str] = {}

        while stack:
            s = stack.pop()
            if s in visited:
                if s == start and parent:
                    # Reconstruct cycle
                    cycle = [s]
                    cur = parent.get(cycle[-1])
                    while cur is not None and cur != start:
                        cycle.append(cur)
                        cur = parent.get(cur)
                    cycle.append(start)
                    cycle.reverse()
                    return cycle
                continue
            visited.add(s)
            for t in self._k.successors(s):
                if t in allowed:
                    if t not in visited or t == start:
                        parent[t] = s
                        stack.append(t)
        return None


# ============================================================================
# CTL* via automata-theoretic approach
# ============================================================================

@dataclass
class TreeAutomatonState:
    """State of the tree automaton used for CTL* model checking."""
    formula: TemporalFormula
    index: int = 0

    def __hash__(self) -> int:
        return hash((str(self.formula), self.index))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TreeAutomatonState):
            return NotImplemented
        return str(self.formula) == str(other.formula) and self.index == other.index


@dataclass
class ObligationSet:
    """A set of temporal obligations on a path.

    Each obligation is an (Until/Release) sub-formula that needs to be
    eventually discharged.  This is the core bookkeeping for the
    automata-theoretic CTL* algorithm.
    """
    pending_until: List[Tuple[TemporalFormula, TemporalFormula]] = field(default_factory=list)
    pending_finally: List[TemporalFormula] = field(default_factory=list)
    required_globally: List[TemporalFormula] = field(default_factory=list)

    def copy(self) -> "ObligationSet":
        return ObligationSet(
            pending_until=list(self.pending_until),
            pending_finally=list(self.pending_finally),
            required_globally=list(self.required_globally),
        )

    def is_empty(self) -> bool:
        return not self.pending_until and not self.pending_finally

    def signature(self) -> Tuple:
        return (
            tuple(sorted(str(u) for _, u in self.pending_until)),
            tuple(sorted(str(f) for f in self.pending_finally)),
            tuple(sorted(str(g) for g in self.required_globally)),
        )


class CTLStarChecker:
    """CTL* model checker integrating CTL labeling, LTL-to-automaton
    reduction, and automata-theoretic tree-automaton approach.

    For the CTL fragment, the efficient labeling algorithm is used directly.
    For full CTL*, the formula is decomposed into maximal state sub-formulas,
    each solved recursively, and path formulas are handled via a product
    construction with a Büchi automaton.
    """

    def __init__(self, coalgebra: object) -> None:
        """Initialize with an FCoalgebra instance."""
        self._coalg = coalgebra
        self._kripke = KripkeAdapter.from_coalgebra(coalgebra)
        self._ctl_labeler = CTLLabeler(self._kripke)
        self._cex_gen = CounterexampleGenerator(self._kripke)

    @property
    def kripke(self) -> KripkeAdapter:
        return self._kripke

    def check(self, formula: TemporalFormula) -> CTLCheckResult:
        """Model-check *formula* on the system.

        Returns a :class:`CTLCheckResult` containing the satisfying
        states and (if the formula is violated) a counterexample.
        """
        logger.info("CTL* checking formula: %s", formula)

        # Normalize
        norm = simplify(to_nnf(formula))

        # Dispatch: use efficient CTL labeling if possible
        if is_ctl(norm):
            sat = self._ctl_labeler.label(norm)
        elif is_ltl(norm):
            sat = self._check_ltl(norm)
        else:
            sat = self._check_ctl_star(norm)

        holds = self._kripke.initial_states <= sat
        cex: Optional[List[str]] = None
        if not holds:
            cex = self._generate_counterexample(norm, sat)

        return CTLCheckResult(
            formula=formula,
            satisfying_states=sat,
            initial_states=self._kripke.initial_states,
            holds=holds,
            counterexample=cex,
        )

    def check_on_quotient(
        self,
        formula: TemporalFormula,
        quotient_coalg: object,
        projection: Mapping[str, str],
    ) -> Tuple[CTLCheckResult, CTLCheckResult]:
        """Check *formula* on both the original system and its quotient.

        Returns (original_result, quotient_result).
        """
        orig_result = self.check(formula)
        q_checker = CTLStarChecker(quotient_coalg)
        q_result = q_checker.check(formula)
        return orig_result, q_result

    # -- LTL fragment -------------------------------------------------------

    def _check_ltl(self, formula: TemporalFormula) -> FrozenSet[str]:
        """Check LTL formula via product with Büchi automaton.

        We convert the LTL formula to a generalized Büchi automaton on-the-fly
        and compute the product with the Kripke structure.  States satisfying
        the formula are those from which the product has an accepting run.
        """
        # Decompose into obligations
        obligations = self._decompose_ltl(formula)

        # For each initial state, simulate the automaton product
        satisfying: Set[str] = set()
        for state in self._kripke.states:
            if self._ltl_accepts_from(state, formula, obligations):
                satisfying.add(state)

        return frozenset(satisfying)

    def _decompose_ltl(self, formula: TemporalFormula) -> ObligationSet:
        """Extract obligations from an LTL formula for automaton construction."""
        obs = ObligationSet()
        self._extract_obligations(formula, obs)
        return obs

    def _extract_obligations(self, f: TemporalFormula, obs: ObligationSet) -> None:
        """Recursively extract Until/Finally/Globally obligations."""
        if isinstance(f, Until):
            obs.pending_until.append((f.left, f.right))
            self._extract_obligations(f.left, obs)
            self._extract_obligations(f.right, obs)
        elif isinstance(f, Finally):
            obs.pending_finally.append(f.child)
            self._extract_obligations(f.child, obs)
        elif isinstance(f, Globally):
            obs.required_globally.append(f.child)
            self._extract_obligations(f.child, obs)
        elif isinstance(f, (And, Or)):
            self._extract_obligations(f.left, obs)
            self._extract_obligations(f.right, obs)
        elif isinstance(f, Not):
            self._extract_obligations(f.child, obs)
        elif isinstance(f, Next):
            self._extract_obligations(f.child, obs)
        elif isinstance(f, Release):
            self._extract_obligations(f.left, obs)
            self._extract_obligations(f.right, obs)

    def _ltl_accepts_from(
        self,
        state: str,
        formula: TemporalFormula,
        obligations: ObligationSet,
    ) -> bool:
        """Check if the formula holds starting from *state* using bounded
        model checking with iterative deepening and cycle detection.

        This is a sound but potentially incomplete approach for finite
        systems.  For full correctness on the finite Kripke structure,
        we check up to |S| steps and look for accepting cycles.
        """
        max_depth = len(self._kripke.states) + 1
        return self._ltl_dfs(state, formula, set(), 0, max_depth)

    def _ltl_dfs(
        self,
        state: str,
        formula: TemporalFormula,
        visited: Set[Tuple[str, str]],
        depth: int,
        max_depth: int,
    ) -> bool:
        """DFS evaluation of LTL formula on concrete paths."""
        if depth > max_depth:
            # Over depth limit – handle terminal cases
            return self._eval_state_formula(state, formula)

        key = (state, str(formula))
        if key in visited:
            # Cycle detected – formula holds on cycle iff it's a globally/release type
            return isinstance(formula, (Globally, TrueFormula))

        visited_copy = visited | {key}

        if isinstance(formula, (Atomic, TrueFormula, FalseFormula)):
            return self._eval_state_formula(state, formula)

        if isinstance(formula, Not):
            if isinstance(formula.child, Atomic):
                return not self._eval_state_formula(state, formula.child)
            return not self._ltl_dfs(state, formula.child, visited_copy, depth, max_depth)

        if isinstance(formula, And):
            return (self._ltl_dfs(state, formula.left, visited_copy, depth, max_depth) and
                    self._ltl_dfs(state, formula.right, visited_copy, depth, max_depth))

        if isinstance(formula, Or):
            return (self._ltl_dfs(state, formula.left, visited_copy, depth, max_depth) or
                    self._ltl_dfs(state, formula.right, visited_copy, depth, max_depth))

        if isinstance(formula, Next):
            for succ in self._kripke.successors(state):
                if self._ltl_dfs(succ, formula.child, visited_copy, depth + 1, max_depth):
                    return True
            return False

        if isinstance(formula, Finally):
            # F φ = φ ∨ X(F φ): check current state or defer
            if self._ltl_dfs(state, formula.child, visited_copy, depth, max_depth):
                return True
            for succ in self._kripke.successors(state):
                if self._ltl_dfs(succ, formula, visited_copy, depth + 1, max_depth):
                    return True
            return False

        if isinstance(formula, Globally):
            # G φ = φ ∧ X(G φ): check current state and all successors
            if not self._ltl_dfs(state, formula.child, visited_copy, depth, max_depth):
                return False
            succs = self._kripke.successors(state)
            if not succs:
                return True  # deadlock state: G φ holds vacuously at end
            for succ in succs:
                if not self._ltl_dfs(succ, formula, visited_copy, depth + 1, max_depth):
                    return False
            return True

        if isinstance(formula, Until):
            # φ U ψ = ψ ∨ (φ ∧ X(φ U ψ))
            if self._ltl_dfs(state, formula.right, visited_copy, depth, max_depth):
                return True
            if not self._ltl_dfs(state, formula.left, visited_copy, depth, max_depth):
                return False
            for succ in self._kripke.successors(state):
                if self._ltl_dfs(succ, formula, visited_copy, depth + 1, max_depth):
                    return True
            return False

        if isinstance(formula, Release):
            # φ R ψ = ψ ∧ (φ ∨ X(φ R ψ))
            if not self._ltl_dfs(state, formula.right, visited_copy, depth, max_depth):
                return False
            if self._ltl_dfs(state, formula.left, visited_copy, depth, max_depth):
                return True
            for succ in self._kripke.successors(state):
                if self._ltl_dfs(succ, formula, visited_copy, depth + 1, max_depth):
                    return True
            return False

        return self._eval_state_formula(state, formula)

    def _eval_state_formula(self, state: str, formula: TemporalFormula) -> bool:
        """Evaluate a state formula at a given state."""
        if isinstance(formula, TrueFormula):
            return True
        if isinstance(formula, FalseFormula):
            return False
        if isinstance(formula, Atomic):
            return self._kripke.has_proposition(state, formula.name)
        if isinstance(formula, Not):
            return not self._eval_state_formula(state, formula.child)
        if isinstance(formula, And):
            return (self._eval_state_formula(state, formula.left) and
                    self._eval_state_formula(state, formula.right))
        if isinstance(formula, Or):
            return (self._eval_state_formula(state, formula.left) or
                    self._eval_state_formula(state, formula.right))
        return state in self._ctl_labeler.label(formula)

    # -- Full CTL* ----------------------------------------------------------

    def _check_ctl_star(self, formula: TemporalFormula) -> FrozenSet[str]:
        """Handle full CTL* formulas by decomposition.

        Strategy:
        1. Find all maximal state sub-formulas.
        2. Label each bottom-up, replacing them with fresh APs.
        3. Handle remaining path formulas via LTL product.
        """
        if formula.is_state_formula():
            return self._ctl_labeler.label(formula)

        # For E(path) or A(path), decompose
        if isinstance(formula, ExistsPath):
            return self._check_exists_path_star(formula.path_formula)
        if isinstance(formula, ForallPath):
            neg = formula.path_formula.negate()
            neg_norm = simplify(to_nnf(neg))
            e_neg = self._check_exists_path_star(neg_norm)
            return self._kripke.states - e_neg

        # For boolean combinations of state formulas
        if isinstance(formula, And):
            return self._check_ctl_star(formula.left) & self._check_ctl_star(formula.right)
        if isinstance(formula, Or):
            return self._check_ctl_star(formula.left) | self._check_ctl_star(formula.right)
        if isinstance(formula, Not):
            return self._kripke.states - self._check_ctl_star(formula.child)

        return self._ctl_labeler.label(formula)

    def _check_exists_path_star(self, path_formula: TemporalFormula) -> FrozenSet[str]:
        """Check E(path_formula) for a general path formula.

        Extracts maximal state sub-formulas from the path formula,
        labels them, introduces fresh atomic propositions, then
        solves the resulting LTL problem on an augmented Kripke structure.
        """
        # Collect maximal state sub-formulas
        state_subs: Dict[str, Tuple[TemporalFormula, FrozenSet[str]]] = {}
        fresh_counter = [0]

        def extract_state_subs(f: TemporalFormula) -> TemporalFormula:
            if f.is_state_formula():
                name = f"__cstar_{fresh_counter[0]}"
                fresh_counter[0] += 1
                sat = self._check_ctl_star(f)
                state_subs[name] = (f, sat)
                return Atomic(name)
            # Recurse into temporal operators
            if isinstance(f, Next):
                return Next(extract_state_subs(f.child))
            if isinstance(f, Finally):
                return Finally(extract_state_subs(f.child))
            if isinstance(f, Globally):
                return Globally(extract_state_subs(f.child))
            if isinstance(f, Until):
                return Until(extract_state_subs(f.left), extract_state_subs(f.right))
            if isinstance(f, Release):
                return Release(extract_state_subs(f.left), extract_state_subs(f.right))
            if isinstance(f, And):
                return And(extract_state_subs(f.left), extract_state_subs(f.right))
            if isinstance(f, Or):
                return Or(extract_state_subs(f.left), extract_state_subs(f.right))
            if isinstance(f, Not):
                return Not(extract_state_subs(f.child))
            return f

        ltl_formula = extract_state_subs(path_formula)

        # Build augmented labels
        aug_labels: Dict[str, FrozenSet[str]] = {}
        for s in self._kripke.states:
            extra: Set[str] = set()
            for name, (_, sat) in state_subs.items():
                if s in sat:
                    extra.add(name)
            aug_labels[s] = self._kripke.labels.get(s, frozenset()) | frozenset(extra)

        # Build augmented Kripke
        aug_kripke = KripkeAdapter(
            states=self._kripke.states,
            initial_states=self._kripke.initial_states,
            transitions=self._kripke.transitions,
            labels=aug_labels,
        )

        # Now check LTL formula on augmented Kripke
        aug_labeler = CTLLabeler(aug_kripke)
        satisfying: Set[str] = set()
        for state in self._kripke.states:
            obs = self._decompose_ltl(ltl_formula)
            if self._ltl_accepts_from(state, ltl_formula, obs):
                satisfying.add(state)

        return frozenset(satisfying)

    # -- Counterexample generation ------------------------------------------

    def _generate_counterexample(
        self,
        formula: TemporalFormula,
        sat_states: FrozenSet[str],
    ) -> Optional[List[str]]:
        """Generate a counterexample trace for a violated formula."""
        # For universal formulas (AG, AF, AU, AX), find path to violation
        if isinstance(formula, ForallPath):
            inner = formula.path_formula
            if isinstance(inner, Globally):
                return self._cex_gen.ag_counterexample(self._ctl_labeler, inner.child)
            if isinstance(inner, Finally):
                # AF φ violated: EG ¬φ witness
                lasso = self._cex_gen.eg_lasso(self._ctl_labeler, Not(inner.child))
                if lasso:
                    stem, cycle = lasso
                    return stem + cycle[1:]
            if isinstance(inner, Until):
                # A[φ U ψ] violated: find path avoiding ψ from initial
                bad_states = self._kripke.states - sat_states
                for init in sorted(self._kripke.initial_states):
                    if init in bad_states:
                        path = self._cex_gen.shortest_path(init, bad_states)
                        if path:
                            return path

        # Generic: find shortest path from an initial state to a violating state
        bad = self._kripke.initial_states - sat_states
        if bad:
            return [next(iter(sorted(bad)))]
        return None


# ============================================================================
# CTL convenience: check common patterns
# ============================================================================

def check_invariant(coalgebra: object, prop: str) -> CTLCheckResult:
    """Check AG(prop) – property holds in all reachable states."""
    from .temporal_logic import AG as _AG
    checker = CTLStarChecker(coalgebra)
    formula = _AG(Atomic(prop))
    return checker.check(formula)


def check_reachability(coalgebra: object, prop: str) -> CTLCheckResult:
    """Check EF(prop) – property is reachable from an initial state."""
    from .temporal_logic import EF as _EF
    checker = CTLStarChecker(coalgebra)
    formula = _EF(Atomic(prop))
    return checker.check(formula)


def check_response(coalgebra: object, trigger: str, response: str) -> CTLCheckResult:
    """Check AG(trigger → AF response) – every trigger leads to response."""
    from .temporal_logic import AG as _AG, AF as _AF
    checker = CTLStarChecker(coalgebra)
    formula = _AG(Implies(Atomic(trigger), _AF(Atomic(response))))
    return checker.check(formula)
