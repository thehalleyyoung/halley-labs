"""
TLA-lite action evaluation.

An *action* is a TLA+ formula that relates the current state to the next
state.  The ``ActionEvaluator`` computes the set of successor states
reachable by a given action from a given state.

Key responsibilities:
* Handle primed (next-state) variable assignments.
* Support conjunctive and disjunctive action composition.
* Support existential quantification within actions.
* Detect stuttering steps and action enablement.
* Compute all initial states from an ``Init`` predicate.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

from .values import (
    TLAValue,
    TLAValueError,
    BoolValue,
    IntValue,
    SetValue,
)
from .environment import Environment
from .state import TLAState, StateSpace
from .evaluator import (
    Expr,
    ExprKind,
    EvalError,
    evaluate,
    _eval,
    _eval_bool,
    _as_set,
)


# ===================================================================
# Action AST helpers
# ===================================================================

@dataclass(frozen=True)
class ActionExpr:
    """Wrapper distinguishing action-level expressions.

    An ``ActionExpr`` is either a plain ``Expr`` evaluated in action mode,
    or one of the compound action forms (conj, disj, exists, unchanged,
    sequential composition).
    """
    kind: str
    data: Any = None
    children: Tuple["ActionExpr", ...] = ()
    expr: Optional[Expr] = None


def action_conj(*actions: ActionExpr) -> ActionExpr:
    return ActionExpr("conj", children=tuple(actions))

def action_disj(*actions: ActionExpr) -> ActionExpr:
    return ActionExpr("disj", children=tuple(actions))

def action_exists(var: str, domain_expr: Expr, body: ActionExpr) -> ActionExpr:
    return ActionExpr("exists", data={"var": var, "domain": domain_expr}, children=(body,))

def action_unchanged(*vars: str) -> ActionExpr:
    return ActionExpr("unchanged", data=tuple(vars))

def action_from_expr(expr: Expr) -> ActionExpr:
    """Wrap a plain Expr as an action (it may reference primed variables)."""
    return ActionExpr("expr", expr=expr)

def action_seq(first: ActionExpr, second: ActionExpr) -> ActionExpr:
    """Sequential composition: first is applied, then second on the result."""
    return ActionExpr("seq", children=(first, second))


# ===================================================================
# Partial assignment – tracks which primed variables have been set
# ===================================================================

class _PartialAssignment:
    """Mutable accumulator for primed variable bindings during action eval."""

    __slots__ = ("_updates", "_constraints")

    def __init__(self) -> None:
        self._updates: Dict[str, TLAValue] = {}
        self._constraints: List[Callable[[Dict[str, TLAValue]], bool]] = []

    def assign(self, var: str, val: TLAValue) -> None:
        if var in self._updates and self._updates[var] != val:
            raise _ConflictError(
                f"Conflicting assignments for {var}': "
                f"{self._updates[var].pretty()} vs {val.pretty()}"
            )
        self._updates[var] = val

    def add_constraint(self, pred: Callable[[Dict[str, TLAValue]], bool]) -> None:
        self._constraints.append(pred)

    def has(self, var: str) -> bool:
        return var in self._updates

    def get(self, var: str) -> Optional[TLAValue]:
        return self._updates.get(var)

    @property
    def updates(self) -> Dict[str, TLAValue]:
        return dict(self._updates)

    def satisfies_constraints(self) -> bool:
        for c in self._constraints:
            if not c(self._updates):
                return False
        return True

    def copy(self) -> "_PartialAssignment":
        pa = _PartialAssignment()
        pa._updates = dict(self._updates)
        pa._constraints = list(self._constraints)
        return pa

    def merge(self, other: "_PartialAssignment") -> "_PartialAssignment":
        """Merge two partial assignments; raises on conflict."""
        result = self.copy()
        for var, val in other._updates.items():
            result.assign(var, val)
        result._constraints.extend(other._constraints)
        return result


class _ConflictError(TLAValueError):
    """Internal: two conjuncts assign conflicting values to a variable."""


# ===================================================================
# ActionEvaluator
# ===================================================================

class ActionEvaluator:
    """Evaluates TLA+ actions to compute successor states.

    The evaluator works by collecting primed-variable assignments from an
    action formula applied to a source state, then constructing successor
    states from the assignments.
    """

    def __init__(self, env: Environment, state_vars: Tuple[str, ...]) -> None:
        self._env = env
        self._state_vars = state_vars

    @property
    def state_vars(self) -> Tuple[str, ...]:
        return self._state_vars

    # --- public API -------------------------------------------------------

    def evaluate_action(self, action: ActionExpr, state: TLAState) -> Set[TLAState]:
        """Return all next-states reachable via *action* from *state*."""
        assignments = self._collect_assignments(action, state)
        successors: Set[TLAState] = set()
        for pa in assignments:
            if pa.satisfies_constraints():
                next_state = self._build_next_state(state, pa)
                if next_state is not None:
                    successors.add(next_state)
        return successors

    def is_enabled(self, action: ActionExpr, state: TLAState) -> bool:
        """Check if *action* is enabled in *state* (has at least one successor)."""
        return len(self.evaluate_action(action, state)) > 0

    def is_stuttering(self, action: ActionExpr, state: TLAState) -> bool:
        """Check if all successors of *action* from *state* equal *state*."""
        successors = self.evaluate_action(action, state)
        return all(s == state for s in successors)

    def evaluate_init(self, init_expr: Expr) -> Set[TLAState]:
        """Compute the set of initial states from an Init predicate.

        The Init predicate is evaluated by enumerating all possible
        variable assignments that make it true.
        """
        return self._compute_init_states(init_expr)

    def next_state_relation(
        self,
        init_expr: Expr,
        next_action: ActionExpr,
        *,
        max_states: int = 100_000,
    ) -> StateSpace:
        """BFS exploration of the state graph defined by Init and Next.

        Returns a ``StateSpace`` containing all reachable states and
        transitions.
        """
        space = StateSpace()
        initial = self.evaluate_init(init_expr)
        if not initial:
            return space

        frontier: List[TLAState] = []
        for s in initial:
            if space.add(s, is_initial=True):
                frontier.append(s)

        while frontier and len(space) < max_states:
            current = frontier.pop(0)
            successors = self.evaluate_action(next_action, current)
            for succ in successors:
                space.add_transition(current, succ)
                if space.add(succ):
                    frontier.append(succ)

        return space

    # --- internal: assignment collection ----------------------------------

    def _collect_assignments(
        self, action: ActionExpr, state: TLAState
    ) -> List[_PartialAssignment]:
        """Recursively collect all valid partial assignments."""

        kind = action.kind

        if kind == "expr":
            return self._collect_from_expr(action.expr, state)

        if kind == "conj":
            return self._collect_conj(action.children, state)

        if kind == "disj":
            return self._collect_disj(action.children, state)

        if kind == "exists":
            return self._collect_exists(action, state)

        if kind == "unchanged":
            return self._collect_unchanged(action.data, state)

        if kind == "seq":
            return self._collect_seq(action.children[0], action.children[1], state)

        raise EvalError(f"Unknown action kind: {kind}")

    def _collect_from_expr(
        self, expr: Expr, state: TLAState
    ) -> List[_PartialAssignment]:
        """Evaluate a plain expression in action context.

        If the expression is a boolean predicate (a guard), we check it.
        If it contains primed variable assignments, we extract them.
        """
        # Detect conjunctive form at expression level
        if expr.kind is ExprKind.BINARY_OP and expr.data in ("/\\", "\\land", "&&"):
            left_results = self._collect_from_expr(expr.children[0], state)
            combined: List[_PartialAssignment] = []
            for lpa in left_results:
                env_copy = self._env.snapshot()
                for var, val in lpa.updates.items():
                    env_copy.bind(var + "'", val)
                right_results = self._collect_from_expr_with_env(
                    expr.children[1], state, env_copy
                )
                for rpa in right_results:
                    try:
                        combined.append(lpa.merge(rpa))
                    except _ConflictError:
                        pass
            return combined

        # Detect disjunctive form at expression level
        if expr.kind is ExprKind.BINARY_OP and expr.data in ("\\/", "\\lor", "||"):
            left_results = self._collect_from_expr(expr.children[0], state)
            right_results = self._collect_from_expr(expr.children[1], state)
            return left_results + right_results

        # Detect existential quantifier in action context
        if expr.kind is ExprKind.QUANT_EXISTS:
            var = expr.data
            domain_expr = expr.children[0]
            body_expr = expr.children[1]
            domain = _as_set(evaluate(domain_expr, self._env, state), "\\E domain in action")
            all_assignments: List[_PartialAssignment] = []
            for elem in domain:
                env_copy = self._env.snapshot()
                env_copy.push_scope(f"exists:{var}")
                env_copy.bind(var, elem)
                body_results = self._collect_from_expr_with_env(body_expr, state, env_copy)
                all_assignments.extend(body_results)
            return all_assignments

        # Detect primed variable equality: x' = expr
        if expr.kind is ExprKind.BINARY_OP and expr.data == "=":
            left_child, right_child = expr.children
            if left_child.kind is ExprKind.PRIMED_REF:
                var_name = left_child.data
                val = evaluate(right_child, self._env, state)
                pa = _PartialAssignment()
                pa.assign(var_name, val)
                return [pa]
            # expr = x' (reversed)
            if right_child.kind is ExprKind.PRIMED_REF:
                var_name = right_child.data
                val = evaluate(left_child, self._env, state)
                pa = _PartialAssignment()
                pa.assign(var_name, val)
                return [pa]

        # Detect UNCHANGED
        if expr.kind is ExprKind.UNCHANGED:
            return self._collect_unchanged(expr.data, state)

        # General expression: evaluate as boolean guard
        env_copy = self._env.snapshot()
        self._bind_primed_from_env(env_copy, state)

        try:
            result = evaluate(expr, env_copy, state)
        except EvalError:
            return []

        if isinstance(result, BoolValue):
            if result.val:
                return [_PartialAssignment()]
            else:
                return []

        return [_PartialAssignment()]

    def _collect_from_expr_with_env(
        self, expr: Expr, state: TLAState, env: Environment
    ) -> List[_PartialAssignment]:
        """Like _collect_from_expr but with a specific environment."""
        saved = self._env
        self._env = env
        try:
            return self._collect_from_expr(expr, state)
        finally:
            self._env = saved

    def _collect_conj(
        self, children: Tuple[ActionExpr, ...], state: TLAState
    ) -> List[_PartialAssignment]:
        """Conjunctive action: all conjuncts must be satisfiable."""
        if not children:
            return [_PartialAssignment()]

        current = self._collect_assignments(children[0], state)
        for child in children[1:]:
            next_round: List[_PartialAssignment] = []
            for existing_pa in current:
                env_copy = self._env.snapshot()
                for var, val in existing_pa.updates.items():
                    env_copy.bind(var + "'", val)
                saved = self._env
                self._env = env_copy
                try:
                    child_results = self._collect_assignments(child, state)
                finally:
                    self._env = saved
                for child_pa in child_results:
                    try:
                        next_round.append(existing_pa.merge(child_pa))
                    except _ConflictError:
                        pass
            current = next_round
        return current

    def _collect_disj(
        self, children: Tuple[ActionExpr, ...], state: TLAState
    ) -> List[_PartialAssignment]:
        """Disjunctive action: collect from all disjuncts."""
        result: List[_PartialAssignment] = []
        for child in children:
            result.extend(self._collect_assignments(child, state))
        return result

    def _collect_exists(
        self, action: ActionExpr, state: TLAState
    ) -> List[_PartialAssignment]:
        """Existential quantification: \\E var \\in S : Action(var)."""
        var = action.data["var"]
        domain_expr = action.data["domain"]
        domain = _as_set(evaluate(domain_expr, self._env, state), "\\E domain")
        result: List[_PartialAssignment] = []
        for elem in domain:
            env_copy = self._env.snapshot()
            env_copy.push_scope(f"exists:{var}")
            env_copy.bind(var, elem)
            saved = self._env
            self._env = env_copy
            try:
                child_results = self._collect_assignments(action.children[0], state)
                result.extend(child_results)
            finally:
                self._env = saved
        return result

    def _collect_unchanged(
        self, var_names: Tuple[str, ...], state: TLAState
    ) -> List[_PartialAssignment]:
        """UNCHANGED <<v1, v2, ...>>: keep listed variables the same."""
        pa = _PartialAssignment()
        for vname in var_names:
            pa.assign(vname, state.get(vname))
        return [pa]

    def _collect_seq(
        self, first: ActionExpr, second: ActionExpr, state: TLAState
    ) -> List[_PartialAssignment]:
        """Sequential composition: apply first, then second on result."""
        first_results = self._collect_assignments(first, state)
        all_results: List[_PartialAssignment] = []
        for fpa in first_results:
            mid_state = self._build_next_state(state, fpa)
            if mid_state is None:
                continue
            second_results = self._collect_assignments(second, mid_state)
            all_results.extend(second_results)
        return all_results

    # --- helpers ----------------------------------------------------------

    def _build_next_state(
        self, current: TLAState, pa: _PartialAssignment
    ) -> Optional[TLAState]:
        """Build a complete next state from a partial assignment.

        Any unassigned state variable retains its current value (implicit
        stuttering on that variable).
        """
        updates = pa.updates
        bindings: Dict[str, TLAValue] = {}
        for var in self._state_vars:
            if var in updates:
                bindings[var] = updates[var]
            elif current.has_var(var):
                bindings[var] = current.get(var)
            else:
                return None
        return TLAState(bindings)

    def _bind_primed_from_env(self, env: Environment, state: TLAState) -> None:
        """Bind primed variables from current env if available."""
        for var in self._state_vars:
            key = var + "'"
            val = env.lookup(key)
            if val is None and state.has_var(var):
                # For guard evaluation without primed bindings, bind to current
                pass

    # --- init state computation -------------------------------------------

    def _compute_init_states(self, init_expr: Expr) -> Set[TLAState]:
        """Evaluate an Init predicate to find all initial states.

        This is done by treating Init as an action on an empty state,
        collecting all primed variable assignments, and treating those
        as the initial bindings.

        For simple Init predicates like ``x = 0 /\\ y = 1``, this
        extracts the direct assignments.
        """
        results = self._extract_init_assignments(init_expr)
        states: Set[TLAState] = set()
        for pa in results:
            bindings = pa.updates
            # Check that all state vars are assigned
            if all(v in bindings for v in self._state_vars):
                states.add(TLAState(bindings))
        return states

    def _extract_init_assignments(self, expr: Expr) -> List[_PartialAssignment]:
        """Extract variable assignments from an Init predicate."""
        empty = TLAState()

        # Conjunction: combine both sides
        if expr.kind is ExprKind.BINARY_OP and expr.data in ("/\\", "\\land"):
            left_res = self._extract_init_assignments(expr.children[0])
            all_results: List[_PartialAssignment] = []
            for lpa in left_res:
                env_copy = self._env.snapshot()
                for var, val in lpa.updates.items():
                    env_copy.bind(var, val)
                saved = self._env
                self._env = env_copy
                try:
                    right_res = self._extract_init_assignments(expr.children[1])
                finally:
                    self._env = saved
                for rpa in right_res:
                    try:
                        all_results.append(lpa.merge(rpa))
                    except _ConflictError:
                        pass
            return all_results

        # Disjunction: union of both sides
        if expr.kind is ExprKind.BINARY_OP and expr.data in ("\\/", "\\lor"):
            left_res = self._extract_init_assignments(expr.children[0])
            right_res = self._extract_init_assignments(expr.children[1])
            return left_res + right_res

        # Equality: x = val (init assigns non-primed variables)
        if expr.kind is ExprKind.BINARY_OP and expr.data == "=":
            left_child, right_child = expr.children
            if left_child.kind is ExprKind.NAME_REF:
                var_name = left_child.data
                if var_name in self._state_vars:
                    try:
                        val = evaluate(right_child, self._env, empty)
                        pa = _PartialAssignment()
                        pa.assign(var_name, val)
                        return [pa]
                    except (EvalError, TLAValueError):
                        pass
            if right_child.kind is ExprKind.NAME_REF:
                var_name = right_child.data
                if var_name in self._state_vars:
                    try:
                        val = evaluate(left_child, self._env, empty)
                        pa = _PartialAssignment()
                        pa.assign(var_name, val)
                        return [pa]
                    except (EvalError, TLAValueError):
                        pass

        # Membership: x \\in S (generates one state per element)
        if expr.kind is ExprKind.BINARY_OP and expr.data in ("\\in", "∈"):
            left_child = expr.children[0]
            right_child = expr.children[1]
            if left_child.kind is ExprKind.NAME_REF:
                var_name = left_child.data
                if var_name in self._state_vars:
                    try:
                        sval = evaluate(right_child, self._env, empty)
                        if isinstance(sval, SetValue):
                            results: List[_PartialAssignment] = []
                            for elem in sval:
                                pa = _PartialAssignment()
                                pa.assign(var_name, elem)
                                results.append(pa)
                            return results
                    except (EvalError, TLAValueError):
                        pass

        # Existential quantifier
        if expr.kind is ExprKind.QUANT_EXISTS:
            var = expr.data
            domain_expr = expr.children[0]
            body_expr = expr.children[1]
            try:
                domain = _as_set(evaluate(domain_expr, self._env, empty), "\\E domain in init")
            except (EvalError, TLAValueError):
                return []
            all_results: List[_PartialAssignment] = []
            for elem in domain:
                env_copy = self._env.snapshot()
                env_copy.push_scope(f"exists:{var}")
                env_copy.bind(var, elem)
                saved = self._env
                self._env = env_copy
                try:
                    body_res = self._extract_init_assignments(body_expr)
                    all_results.extend(body_res)
                finally:
                    self._env = saved
            return all_results

        # Fallback: try evaluating as a guard
        return [_PartialAssignment()]


# ===================================================================
# Convenience functions
# ===================================================================

def compute_successors(
    action: ActionExpr,
    state: TLAState,
    env: Environment,
    state_vars: Tuple[str, ...],
) -> Set[TLAState]:
    """Compute successor states for *action* from *state*."""
    ae = ActionEvaluator(env, state_vars)
    return ae.evaluate_action(action, state)


def compute_initial_states(
    init_expr: Expr,
    env: Environment,
    state_vars: Tuple[str, ...],
) -> Set[TLAState]:
    """Compute the set of initial states."""
    ae = ActionEvaluator(env, state_vars)
    return ae.evaluate_init(init_expr)


def explore_state_space(
    init_expr: Expr,
    next_action: ActionExpr,
    env: Environment,
    state_vars: Tuple[str, ...],
    *,
    max_states: int = 100_000,
) -> StateSpace:
    """Explore the entire state space via BFS."""
    ae = ActionEvaluator(env, state_vars)
    return ae.next_state_relation(init_expr, next_action, max_states=max_states)


def is_action_enabled(
    action: ActionExpr,
    state: TLAState,
    env: Environment,
    state_vars: Tuple[str, ...],
) -> bool:
    """Check if *action* is enabled in *state*."""
    ae = ActionEvaluator(env, state_vars)
    return ae.is_enabled(action, state)


def detect_stuttering(
    action: ActionExpr,
    state: TLAState,
    env: Environment,
    state_vars: Tuple[str, ...],
) -> bool:
    """Check if *action* produces only stuttering steps from *state*."""
    ae = ActionEvaluator(env, state_vars)
    return ae.is_stuttering(action, state)
