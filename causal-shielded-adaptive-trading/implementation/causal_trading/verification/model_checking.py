"""
Symbolic model checking over finite MDP abstractions.

Provides bounded model checking for safety properties, reachability analysis,
probabilistic model checking (PCTL-style), value iteration for probability
computation, counter-example generation, and witness generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


# ---------------------------------------------------------------------------
# Data structures for MDP / specifications
# ---------------------------------------------------------------------------

class SpecKind(Enum):
    """Kind of specification to check."""
    SAFETY = auto()          # "always safe"
    REACHABILITY = auto()    # "eventually reach target"
    PROB_SAFETY = auto()     # Pr[ G safe ] >= threshold
    PROB_REACH = auto()      # Pr[ F target ] >= threshold
    REWARD_BOUND = auto()    # E[total reward] <= bound


@dataclass
class Specification:
    """A temporal specification to model-check."""
    kind: SpecKind
    safe_states: Optional[FrozenSet[int]] = None
    target_states: Optional[FrozenSet[int]] = None
    threshold: float = 0.0
    horizon: int = 100
    reward_bound: float = 0.0
    label: str = ""

    def __repr__(self) -> str:
        return f"Specification({self.kind.name}, horizon={self.horizon}, threshold={self.threshold})"


@dataclass
class MDPTransition:
    """Sparse transition structure for a single (state, action) pair."""
    next_states: NDArray   # int array of successor states
    probs: NDArray         # corresponding probabilities
    reward: float = 0.0

    def expected_reward(self) -> float:
        return self.reward


@dataclass
class MDP:
    """
    Tabular Markov Decision Process.

    Parameters
    ----------
    n_states : int
    n_actions : int
    transitions : dict mapping (state, action) -> MDPTransition
    initial_state : int
    """
    n_states: int
    n_actions: int
    transitions: Dict[Tuple[int, int], MDPTransition]
    initial_state: int = 0
    state_labels: Dict[int, Set[str]] = field(default_factory=dict)

    def available_actions(self, state: int) -> List[int]:
        """Return actions available in *state*."""
        return [a for a in range(self.n_actions) if (state, a) in self.transitions]

    def get_transition_matrix(self, action: int) -> sparse.csr_matrix:
        """Build sparse transition matrix for a fixed action."""
        rows, cols, data = [], [], []
        for s in range(self.n_states):
            t = self.transitions.get((s, action))
            if t is None:
                rows.append(s)
                cols.append(s)
                data.append(1.0)
            else:
                for ns, p in zip(t.next_states, t.probs):
                    rows.append(s)
                    cols.append(int(ns))
                    data.append(float(p))
        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.n_states, self.n_states)
        )

    def reward_vector(self, action: int) -> NDArray:
        """Reward vector for a fixed action."""
        r = np.zeros(self.n_states)
        for s in range(self.n_states):
            t = self.transitions.get((s, action))
            if t is not None:
                r[s] = t.reward
        return r


@dataclass
class CounterExample:
    """A counter-example trace witnessing specification violation."""
    states: List[int]
    actions: List[int]
    violation_step: int
    violation_type: str
    probability: float = 0.0

    def __repr__(self) -> str:
        return (
            f"CounterExample(len={len(self.states)}, "
            f"violation_step={self.violation_step}, "
            f"type={self.violation_type})"
        )


@dataclass
class Witness:
    """A witness trace showing specification satisfaction."""
    states: List[int]
    actions: List[int]
    satisfaction_step: int
    probability: float = 0.0


@dataclass
class CheckResult:
    """Result of a model-checking query."""
    satisfied: bool
    satisfaction_prob: float
    per_state_prob: NDArray              # probability at each state
    counter_examples: List[CounterExample] = field(default_factory=list)
    witnesses: List[Witness] = field(default_factory=list)
    n_iterations: int = 0
    policy: Optional[NDArray] = None     # (n_states,) best action per state

    def __repr__(self) -> str:
        return (
            f"CheckResult(satisfied={self.satisfied}, "
            f"prob={self.satisfaction_prob:.6f}, "
            f"counter_examples={len(self.counter_examples)})"
        )


# ---------------------------------------------------------------------------
# SymbolicModelChecker
# ---------------------------------------------------------------------------

class SymbolicModelChecker:
    """
    Symbolic model checker for finite MDPs.

    Supports bounded safety, reachability, and PCTL-style probabilistic
    properties.  Uses value iteration with sparse matrix representation.

    Parameters
    ----------
    convergence_tol : float
        Tolerance for value iteration convergence.
    max_iterations : int
        Cap on value-iteration sweeps.
    """

    def __init__(
        self,
        convergence_tol: float = 1e-10,
        max_iterations: int = 10000,
    ) -> None:
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check(
        self,
        mdp: MDP,
        spec: Specification,
        horizon: Optional[int] = None,
    ) -> CheckResult:
        """
        Model-check *spec* on *mdp*.

        Parameters
        ----------
        mdp : MDP
        spec : Specification
        horizon : int, optional
            Override the specification horizon.

        Returns
        -------
        CheckResult
        """
        h = horizon if horizon is not None else spec.horizon

        if spec.kind == SpecKind.SAFETY:
            return self._check_safety(mdp, spec, h)
        elif spec.kind == SpecKind.REACHABILITY:
            return self._check_reachability(mdp, spec, h)
        elif spec.kind == SpecKind.PROB_SAFETY:
            return self._check_prob_safety(mdp, spec, h)
        elif spec.kind == SpecKind.PROB_REACH:
            return self._check_prob_reach(mdp, spec, h)
        elif spec.kind == SpecKind.REWARD_BOUND:
            return self._check_reward_bound(mdp, spec, h)
        else:
            raise ValueError(f"Unknown spec kind: {spec.kind}")

    # ------------------------------------------------------------------
    # Satisfaction probability per state
    # ------------------------------------------------------------------

    def compute_satisfaction_probability(
        self,
        mdp: MDP,
        spec: Specification,
        state: int,
        action: Optional[int] = None,
    ) -> float:
        """
        Compute the probability of satisfying *spec* from (state, action).

        If *action* is None, maximises over available actions.
        """
        result = self.check(mdp, spec)
        if action is not None:
            t = mdp.transitions.get((state, action))
            if t is None:
                return 0.0
            return float(np.dot(t.probs, result.per_state_prob[t.next_states]))
        return float(result.per_state_prob[state])

    # ------------------------------------------------------------------
    # Multi-vertex verification
    # ------------------------------------------------------------------

    def verify_all_vertices(
        self,
        base_mdp: MDP,
        spec: Specification,
        vertex_matrices: NDArray,
        horizon: Optional[int] = None,
    ) -> Tuple[bool, List[CheckResult]]:
        """
        Verify *spec* on the MDP parameterised by each vertex transition
        matrix in *vertex_matrices*.

        Parameters
        ----------
        base_mdp : MDP
            Template MDP (structure only; transitions will be overwritten).
        spec : Specification
        vertex_matrices : NDArray, shape (n_vertices, n_states, n_states)
            Each slice is a transition matrix.
        horizon : int, optional

        Returns
        -------
        all_sat : bool
            True iff every vertex satisfies the spec.
        results : list of CheckResult
        """
        results: List[CheckResult] = []
        all_sat = True
        for v_idx in range(vertex_matrices.shape[0]):
            mdp_v = self._instantiate_mdp(base_mdp, vertex_matrices[v_idx])
            res = self.check(mdp_v, spec, horizon)
            results.append(res)
            if not res.satisfied:
                all_sat = False
        return all_sat, results

    # ------------------------------------------------------------------
    # Safety checking
    # ------------------------------------------------------------------

    def _check_safety(self, mdp: MDP, spec: Specification, horizon: int) -> CheckResult:
        """Bounded safety: Pr[ G[0,H] safe ] under maximising scheduler."""
        safe = spec.safe_states or frozenset(range(mdp.n_states))
        n = mdp.n_states

        # prob[s] = max-probability of staying in safe for remaining steps
        prob = np.zeros(n)
        for s in range(n):
            prob[s] = 1.0 if s in safe else 0.0

        policy = np.zeros(n, dtype=int)
        n_iter = 0

        for t in range(horizon):
            new_prob = np.zeros(n)
            for s in range(n):
                if s not in safe:
                    new_prob[s] = 0.0
                    continue
                best_p = 0.0
                best_a = 0
                for a in mdp.available_actions(s):
                    tr = mdp.transitions[(s, a)]
                    p = float(np.dot(tr.probs, prob[tr.next_states]))
                    if p > best_p:
                        best_p = p
                        best_a = a
                new_prob[s] = best_p
                policy[s] = best_a
            prob = new_prob
            n_iter += 1

        sat_prob = float(prob[mdp.initial_state])
        satisfied = sat_prob >= spec.threshold if spec.threshold > 0 else sat_prob >= 1.0 - 1e-9

        cexs = self._generate_safety_counterexamples(mdp, safe, policy, horizon) if not satisfied else []
        return CheckResult(
            satisfied=satisfied,
            satisfaction_prob=sat_prob,
            per_state_prob=prob,
            counter_examples=cexs,
            n_iterations=n_iter,
            policy=policy,
        )

    # ------------------------------------------------------------------
    # Reachability checking
    # ------------------------------------------------------------------

    def _check_reachability(self, mdp: MDP, spec: Specification, horizon: int) -> CheckResult:
        """Bounded reachability: Pr[ F[0,H] target ] under maximising scheduler."""
        target = spec.target_states or frozenset()
        n = mdp.n_states

        prob = np.zeros(n)
        for s in target:
            if s < n:
                prob[s] = 1.0

        policy = np.zeros(n, dtype=int)
        n_iter = 0

        for t in range(horizon):
            new_prob = np.zeros(n)
            for s in range(n):
                if s in target:
                    new_prob[s] = 1.0
                    continue
                best_p = 0.0
                best_a = 0
                for a in mdp.available_actions(s):
                    tr = mdp.transitions[(s, a)]
                    p = float(np.dot(tr.probs, prob[tr.next_states]))
                    if p > best_p:
                        best_p = p
                        best_a = a
                new_prob[s] = best_p
                policy[s] = best_a
            prob = new_prob
            n_iter += 1

        sat_prob = float(prob[mdp.initial_state])
        satisfied = sat_prob >= spec.threshold

        witnesses = self._generate_reachability_witnesses(mdp, target, policy, horizon) if satisfied else []
        cexs = [] if satisfied else self._generate_reachability_counterexamples(mdp, target, policy, horizon)
        return CheckResult(
            satisfied=satisfied,
            satisfaction_prob=sat_prob,
            per_state_prob=prob,
            counter_examples=cexs,
            witnesses=witnesses,
            n_iterations=n_iter,
            policy=policy,
        )

    # ------------------------------------------------------------------
    # Probabilistic safety (PCTL)
    # ------------------------------------------------------------------

    def _check_prob_safety(self, mdp: MDP, spec: Specification, horizon: int) -> CheckResult:
        """PCTL-style:  P>=threshold [ G[0,H] safe ]."""
        result = self._check_safety(mdp, spec, horizon)
        result.satisfied = result.satisfaction_prob >= spec.threshold
        return result

    # ------------------------------------------------------------------
    # Probabilistic reachability (PCTL)
    # ------------------------------------------------------------------

    def _check_prob_reach(self, mdp: MDP, spec: Specification, horizon: int) -> CheckResult:
        """PCTL-style:  P>=threshold [ F[0,H] target ]."""
        result = self._check_reachability(mdp, spec, horizon)
        result.satisfied = result.satisfaction_prob >= spec.threshold
        return result

    # ------------------------------------------------------------------
    # Reward bound checking
    # ------------------------------------------------------------------

    def _check_reward_bound(self, mdp: MDP, spec: Specification, horizon: int) -> CheckResult:
        """Check E[total reward up to H] <= bound under minimising scheduler."""
        n = mdp.n_states
        values = np.zeros(n)
        policy = np.zeros(n, dtype=int)
        n_iter = 0

        for t in range(horizon):
            new_values = np.full(n, np.inf)
            for s in range(n):
                best_v = np.inf
                best_a = 0
                for a in mdp.available_actions(s):
                    tr = mdp.transitions[(s, a)]
                    v = tr.reward + float(np.dot(tr.probs, values[tr.next_states]))
                    if v < best_v:
                        best_v = v
                        best_a = a
                if best_v == np.inf:
                    best_v = 0.0
                new_values[s] = best_v
                policy[s] = best_a
            values = new_values
            n_iter += 1

        total = float(values[mdp.initial_state])
        satisfied = total <= spec.reward_bound
        return CheckResult(
            satisfied=satisfied,
            satisfaction_prob=1.0 if satisfied else 0.0,
            per_state_prob=values,
            n_iterations=n_iter,
            policy=policy,
        )

    # ------------------------------------------------------------------
    # Value iteration (sparse)
    # ------------------------------------------------------------------

    def value_iteration_sparse(
        self,
        mdp: MDP,
        target: FrozenSet[int],
        horizon: int,
        maximize: bool = True,
    ) -> Tuple[NDArray, NDArray]:
        """
        Sparse value iteration for bounded reachability.

        Returns
        -------
        prob : NDArray, shape (n_states,)
        policy : NDArray, shape (n_states,)
        """
        n = mdp.n_states
        prob = np.zeros(n)
        for s in target:
            if s < n:
                prob[s] = 1.0

        policy = np.zeros(n, dtype=int)
        action_matrices = {}
        for a in range(mdp.n_actions):
            action_matrices[a] = mdp.get_transition_matrix(a)

        for _ in range(horizon):
            new_prob = np.zeros(n)
            for s in range(n):
                if s in target:
                    new_prob[s] = 1.0
                    continue
                actions = mdp.available_actions(s)
                if not actions:
                    continue
                vals = []
                for a in actions:
                    mat = action_matrices[a]
                    row = mat.getrow(s)
                    p = float(row.dot(prob.reshape(-1, 1))[0, 0])
                    vals.append((p, a))
                if maximize:
                    best_p, best_a = max(vals, key=lambda x: x[0])
                else:
                    best_p, best_a = min(vals, key=lambda x: x[0])
                new_prob[s] = best_p
                policy[s] = best_a
            prob = new_prob

        return prob, policy

    # ------------------------------------------------------------------
    # Unbounded value iteration
    # ------------------------------------------------------------------

    def value_iteration_unbounded(
        self,
        mdp: MDP,
        target: FrozenSet[int],
        maximize: bool = True,
    ) -> Tuple[NDArray, NDArray]:
        """
        Unbounded reachability value iteration until convergence.

        Returns (prob, policy).
        """
        n = mdp.n_states
        prob = np.zeros(n)
        for s in target:
            if s < n:
                prob[s] = 1.0

        policy = np.zeros(n, dtype=int)

        for iteration in range(self.max_iterations):
            new_prob = np.zeros(n)
            for s in range(n):
                if s in target:
                    new_prob[s] = 1.0
                    continue
                actions = mdp.available_actions(s)
                if not actions:
                    continue
                vals = []
                for a in actions:
                    tr = mdp.transitions[(s, a)]
                    p = float(np.dot(tr.probs, prob[tr.next_states]))
                    vals.append((p, a))
                if maximize:
                    best_p, best_a = max(vals, key=lambda x: x[0])
                else:
                    best_p, best_a = min(vals, key=lambda x: x[0])
                new_prob[s] = best_p
                policy[s] = best_a

            if np.max(np.abs(new_prob - prob)) < self.convergence_tol:
                prob = new_prob
                break
            prob = new_prob

        return prob, policy

    # ------------------------------------------------------------------
    # Counter-example generation
    # ------------------------------------------------------------------

    def _generate_safety_counterexamples(
        self,
        mdp: MDP,
        safe: FrozenSet[int],
        policy: NDArray,
        horizon: int,
        n_traces: int = 5,
    ) -> List[CounterExample]:
        """Generate counter-example traces that leave the safe set."""
        cexs: List[CounterExample] = []
        rng = np.random.default_rng(0)

        for _ in range(n_traces * 10):
            if len(cexs) >= n_traces:
                break
            states = [mdp.initial_state]
            actions: List[int] = []
            violated = False
            for t in range(horizon):
                s = states[-1]
                if s not in safe:
                    cexs.append(CounterExample(
                        states=list(states),
                        actions=list(actions),
                        violation_step=t,
                        violation_type="left_safe_set",
                    ))
                    violated = True
                    break
                a = int(policy[s])
                tr = mdp.transitions.get((s, a))
                if tr is None:
                    break
                ns = rng.choice(tr.next_states, p=tr.probs)
                states.append(int(ns))
                actions.append(a)
            if not violated and states[-1] not in safe:
                cexs.append(CounterExample(
                    states=list(states),
                    actions=list(actions),
                    violation_step=len(states) - 1,
                    violation_type="left_safe_set",
                ))
        return cexs

    def _generate_reachability_counterexamples(
        self,
        mdp: MDP,
        target: FrozenSet[int],
        policy: NDArray,
        horizon: int,
        n_traces: int = 5,
    ) -> List[CounterExample]:
        """Generate traces that fail to reach the target."""
        cexs: List[CounterExample] = []
        rng = np.random.default_rng(1)

        for _ in range(n_traces * 10):
            if len(cexs) >= n_traces:
                break
            states = [mdp.initial_state]
            actions_trace: List[int] = []
            reached = False
            for t in range(horizon):
                s = states[-1]
                if s in target:
                    reached = True
                    break
                a = int(policy[s])
                tr = mdp.transitions.get((s, a))
                if tr is None:
                    break
                ns = rng.choice(tr.next_states, p=tr.probs)
                states.append(int(ns))
                actions_trace.append(a)
            if not reached:
                cexs.append(CounterExample(
                    states=list(states),
                    actions=list(actions_trace),
                    violation_step=len(states) - 1,
                    violation_type="target_not_reached",
                ))
        return cexs

    # ------------------------------------------------------------------
    # Witness generation
    # ------------------------------------------------------------------

    def _generate_reachability_witnesses(
        self,
        mdp: MDP,
        target: FrozenSet[int],
        policy: NDArray,
        horizon: int,
        n_traces: int = 5,
    ) -> List[Witness]:
        """Generate witness traces that reach the target."""
        witnesses: List[Witness] = []
        rng = np.random.default_rng(2)

        for _ in range(n_traces * 20):
            if len(witnesses) >= n_traces:
                break
            states = [mdp.initial_state]
            actions_trace: List[int] = []
            for t in range(horizon):
                s = states[-1]
                if s in target:
                    witnesses.append(Witness(
                        states=list(states),
                        actions=list(actions_trace),
                        satisfaction_step=t,
                    ))
                    break
                a = int(policy[s])
                tr = mdp.transitions.get((s, a))
                if tr is None:
                    break
                ns = rng.choice(tr.next_states, p=tr.probs)
                states.append(int(ns))
                actions_trace.append(a)
        return witnesses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _instantiate_mdp(template: MDP, transition_matrix: NDArray) -> MDP:
        """
        Create a new MDP with the same structure as *template* but
        transitions from *transition_matrix* (n_states × n_states),
        applied to every action identically.
        """
        n = template.n_states
        n_a = template.n_actions
        new_trans: Dict[Tuple[int, int], MDPTransition] = {}
        for s in range(n):
            row = transition_matrix[s]
            nonzero = np.where(row > 1e-15)[0]
            if len(nonzero) == 0:
                nonzero = np.array([s])
                probs = np.array([1.0])
            else:
                probs = row[nonzero]
                probs /= probs.sum()
            for a in range(n_a):
                old_t = template.transitions.get((s, a))
                r = old_t.reward if old_t is not None else 0.0
                new_trans[(s, a)] = MDPTransition(
                    next_states=nonzero.copy(),
                    probs=probs.copy(),
                    reward=r,
                )
        return MDP(
            n_states=n,
            n_actions=n_a,
            transitions=new_trans,
            initial_state=template.initial_state,
            state_labels=template.state_labels,
        )

    # ------------------------------------------------------------------
    # Strongly connected component analysis (Tarjan)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sccs(mdp: MDP, action: int) -> List[List[int]]:
        """
        Compute strongly connected components of the MDP graph
        under a fixed action (Tarjan's algorithm).
        """
        n = mdp.n_states
        index_counter = [0]
        stack: List[int] = []
        on_stack = [False] * n
        indices = [-1] * n
        lowlinks = [-1] * n
        sccs: List[List[int]] = []

        def strongconnect(v: int) -> None:
            indices[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            tr = mdp.transitions.get((v, action))
            if tr is not None:
                for w in tr.next_states:
                    w = int(w)
                    if indices[w] == -1:
                        strongconnect(w)
                        lowlinks[v] = min(lowlinks[v], lowlinks[w])
                    elif on_stack[w]:
                        lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                scc: List[int] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in range(n):
            if indices[v] == -1:
                strongconnect(v)
        return sccs

    # ------------------------------------------------------------------
    # Predecessor computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_predecessors(mdp: MDP) -> Dict[int, Set[Tuple[int, int]]]:
        """
        Build a predecessor map: for each state s, return the set of
        (state, action) pairs that can transition into s.
        """
        preds: Dict[int, Set[Tuple[int, int]]] = {s: set() for s in range(mdp.n_states)}
        for (s, a), tr in mdp.transitions.items():
            for ns in tr.next_states:
                preds[int(ns)].add((s, a))
        return preds

    @staticmethod
    def backward_reachable(
        mdp: MDP,
        target: FrozenSet[int],
    ) -> Set[int]:
        """
        Compute the set of states from which *target* is reachable
        under some scheduler (backward BFS).
        """
        preds = SymbolicModelChecker.compute_predecessors(mdp)
        visited: Set[int] = set(target)
        frontier = list(target)
        while frontier:
            s = frontier.pop()
            for (ps, _) in preds.get(s, set()):
                if ps not in visited:
                    visited.add(ps)
                    frontier.append(ps)
        return visited


# ---------------------------------------------------------------------------
# Helper: build MDP from transition matrix
# ---------------------------------------------------------------------------

def build_mdp_from_matrix(
    transition_matrix: NDArray,
    n_actions: int = 1,
    rewards: Optional[NDArray] = None,
) -> MDP:
    """
    Construct an MDP from a transition matrix.

    Parameters
    ----------
    transition_matrix : NDArray, shape (n_states, n_states)
    n_actions : int
    rewards : NDArray, shape (n_states,), optional

    Returns
    -------
    MDP
    """
    n = transition_matrix.shape[0]
    if rewards is None:
        rewards = np.zeros(n)
    transitions: Dict[Tuple[int, int], MDPTransition] = {}
    for s in range(n):
        row = transition_matrix[s]
        nonzero = np.where(row > 1e-15)[0]
        if len(nonzero) == 0:
            nonzero = np.array([s])
            probs = np.array([1.0])
        else:
            probs = row[nonzero]
            probs /= probs.sum()
        for a in range(n_actions):
            transitions[(s, a)] = MDPTransition(
                next_states=nonzero.copy(),
                probs=probs.copy(),
                reward=float(rewards[s]),
            )
    return MDP(n_states=n, n_actions=n_actions, transitions=transitions)


def build_product_mdp(mdp: MDP, automaton_states: int, automaton_transitions: Dict) -> MDP:
    """
    Build the product MDP of *mdp* with a finite automaton.

    The automaton is specified by:
    - automaton_states : number of automaton states
    - automaton_transitions : dict mapping (q, mdp_state) -> q'

    Product state space: (mdp_state, automaton_state).
    """
    n_prod = mdp.n_states * automaton_states
    prod_trans: Dict[Tuple[int, int], MDPTransition] = {}

    def encode(s: int, q: int) -> int:
        return s * automaton_states + q

    for s in range(mdp.n_states):
        for q in range(automaton_states):
            prod_s = encode(s, q)
            for a in mdp.available_actions(s):
                tr = mdp.transitions[(s, a)]
                next_prod = []
                next_probs = []
                for ns, p in zip(tr.next_states, tr.probs):
                    ns = int(ns)
                    q_next = automaton_transitions.get((q, ns), q)
                    next_prod.append(encode(ns, q_next))
                    next_probs.append(p)
                prod_trans[(prod_s, a)] = MDPTransition(
                    next_states=np.array(next_prod, dtype=int),
                    probs=np.array(next_probs),
                    reward=tr.reward,
                )

    return MDP(
        n_states=n_prod,
        n_actions=mdp.n_actions,
        transitions=prod_trans,
        initial_state=encode(mdp.initial_state, 0),
    )
