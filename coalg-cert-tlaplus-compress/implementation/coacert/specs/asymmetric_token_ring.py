"""Asymmetric Token Ring specification in TLA-lite AST form.

Models a token-passing ring with ASYMMETRIC roles:
  * Node 0 is the "initiator" — it generates the token and has
    different behavior from other nodes.
  * Nodes 1..N-1 are "relayers" — they forward the token clockwise.
  * The initiator can inject a "priority request" that preempts normal
    token passing, creating asymmetry in the fairness structure.

This benchmark is specifically designed to test coalgebraic compression
on protocols WITHOUT full participant symmetry. Unlike 2PC or leader
election where all participants are interchangeable, here node 0 has
a distinguished role, breaking the permutation symmetry.

Safety:   at most one node holds the token at any time.
Liveness: under fairness, every node eventually receives the token.
Priority: the initiator's priority requests are eventually served.

The expected compression ratio is lower than symmetric protocols because
the initiator cannot be merged with relayers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..parser.ast_nodes import (
    Expression,
    IfThenElse,
    Module,
    Operator,
    OperatorApplication,
    OperatorDef,
    Property,
)
from .spec_utils import (
    ModuleBuilder,
    bool_lit,
    ident,
    int_lit,
    make_conjunction,
    make_disjunction,
    make_eq,
    make_except,
    make_exists_single,
    make_forall_single,
    make_func_apply,
    make_function_construction,
    make_guard,
    make_implies,
    make_in,
    make_int_range,
    make_invariant_property,
    make_ite,
    make_land,
    make_liveness_property,
    make_lnot,
    make_lor,
    make_mod,
    make_neq,
    make_plus,
    make_primed_eq,
    make_primed_func_update,
    make_safety_property,
    make_set_enum,
    make_spec_with_fairness,
    make_temporal_property,
    make_unchanged,
    make_vars_tuple,
    make_wf,
    primed,
    str_lit,
)


_ALL_VARIABLES = ("token", "hasToken", "priorityPending", "served")


class AsymmetricTokenRingSpec:
    """Programmatic builder for an asymmetric token ring TLA-lite spec.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes (>= 3). Node 0 is the initiator.
    """

    def __init__(self, n_nodes: int = 4) -> None:
        if n_nodes < 3:
            raise ValueError("Need at least 3 nodes for asymmetric token ring")
        self._n = n_nodes
        self._module: Optional[Module] = None

    def get_spec(self) -> Module:
        """Return the TLA-lite Module AST."""
        if self._module is None:
            self._module = self._build_module()
        return self._module

    def get_properties(self) -> List[Property]:
        """Return properties to model-check."""
        return [
            self._build_mutex_property(),
            self._build_liveness_property(),
            self._build_priority_property(),
        ]

    def get_config(self, n_nodes: Optional[int] = None) -> Dict[str, Any]:
        n = n_nodes or self._n
        return {
            "spec_name": "AsymmetricTokenRing",
            "n_nodes": n,
            "constants": {"N": n, "Nodes": list(range(n))},
            "invariants": ["Mutex"],
            "properties": ["AllServed", "PriorityServed"],
            "symmetry_sets": [],  # NO symmetry — that's the point
            "state_constraint": None,
            "expected_states": self._estimate_states(n),
        }

    def validate(self) -> List[str]:
        errors: List[str] = []
        spec = self.get_spec()
        if not spec.name:
            errors.append("Module name is empty")
        var_names = set()
        for vd in spec.variables:
            var_names.update(vd.names)
        for expected in _ALL_VARIABLES:
            if expected not in var_names:
                errors.append(f"Missing variable: {expected}")
        def_names = {d.name for d in spec.definitions
                     if isinstance(d, OperatorDef)}
        for expected_def in ("Init", "Next", "PassToken", "InitiatorRequest",
                             "Mutex", "Spec"):
            if expected_def not in def_names:
                errors.append(f"Missing definition: {expected_def}")
        return errors

    @staticmethod
    def supported_configurations() -> List[Dict[str, Any]]:
        return [
            {"n_nodes": 3},
            {"n_nodes": 4},
            {"n_nodes": 5},
            {"n_nodes": 6},
        ]

    # ------------------------------------------------------------------
    # Module construction
    # ------------------------------------------------------------------

    def _build_module(self) -> Module:
        mb = ModuleBuilder("AsymmetricTokenRing")
        mb.add_extends("Naturals")

        mb.add_constants("N", "Nodes")
        mb.add_variables(*_ALL_VARIABLES)

        # Init
        mb.add_definition("Init", self._build_init())

        # Actions
        mb.add_definition("PassToken", self._build_pass_token(), params=["i"])
        mb.add_definition("InitiatorRequest", self._build_initiator_request())
        mb.add_definition("ReceiveToken", self._build_receive_token(), params=["i"])

        # Next
        mb.add_definition("Next", self._build_next())

        # Invariants
        mb.add_definition("Mutex", self._build_mutex())

        # Spec with fairness
        vars_t = make_vars_tuple(*_ALL_VARIABLES)
        fairness = self._build_fairness(vars_t)
        spec_expr = make_spec_with_fairness("Init", "Next", vars_t, fairness)
        mb.add_definition("Spec", spec_expr)

        # Liveness definitions
        mb.add_definition("AllServed", self._build_all_served())
        mb.add_definition("PriorityServed", self._build_priority_served())

        for prop in self.get_properties():
            mb.add_property(prop)

        return mb.build()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _build_init(self) -> Expression:
        """Initial state: node 0 has the token, no priority pending."""
        # token = 0
        token_init = make_eq(ident("token"), int_lit(0))
        # hasToken = [i ∈ Nodes ↦ IF i = 0 THEN TRUE ELSE FALSE]
        has_token_init = make_eq(
            ident("hasToken"),
            make_function_construction(
                "i", ident("Nodes"),
                make_ite(
                    make_eq(ident("i"), int_lit(0)),
                    bool_lit(True),
                    bool_lit(False),
                ),
            ),
        )
        # priorityPending = FALSE
        priority_init = make_eq(ident("priorityPending"), bool_lit(False))
        # served = [i ∈ Nodes ↦ FALSE]
        served_init = make_eq(
            ident("served"),
            make_function_construction(
                "i", ident("Nodes"), bool_lit(False),
            ),
        )
        return make_conjunction(
            [token_init, has_token_init, priority_init, served_init]
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _build_pass_token(self) -> Expression:
        """PassToken(i): node i passes the token to (i+1) mod N.

        If i = 0 (initiator) and priorityPending, skip normal passing
        and instead serve the priority request first.
        """
        # Guard: hasToken[i] = TRUE
        guard = make_eq(
            make_func_apply(ident("hasToken"), ident("i")),
            bool_lit(True),
        )
        # next_node = (i + 1) % N
        next_node = make_mod(
            make_plus(ident("i"), int_lit(1)),
            ident("N"),
        )
        # hasToken' = [hasToken EXCEPT ![i] = FALSE, ![next_node] = TRUE]
        token_update = make_conjunction([
            make_primed_func_update("hasToken", ident("i"), bool_lit(False)),
            make_primed_func_update("hasToken", next_node, bool_lit(True)),
        ])
        # token' = next_node
        token_move = make_primed_eq("token", next_node)
        # served' = [served EXCEPT ![next_node] = TRUE]
        served_update = make_primed_func_update(
            "served", next_node, bool_lit(True)
        )
        unchanged_prio = make_unchanged("priorityPending")

        return make_guard(
            guard,
            [token_move, token_update, served_update],
            ["priorityPending"],
        )

    def _build_initiator_request(self) -> Expression:
        """InitiatorRequest: node 0 raises a priority request.

        This is the asymmetric action — only node 0 can do this.
        """
        # Guard: token ≠ 0 (initiator doesn't have token)
        guard = make_neq(ident("token"), int_lit(0))
        # priorityPending' = TRUE
        prio_set = make_primed_eq("priorityPending", bool_lit(True))
        unchanged = make_conjunction([
            make_unchanged("token"),
            make_unchanged("hasToken"),
            make_unchanged("served"),
        ])
        return make_guard(guard, [prio_set], ["token", "hasToken", "served"])

    def _build_receive_token(self) -> Expression:
        """ReceiveToken(i): node i receives the token.

        If i = 0 and priorityPending, clear the priority flag.
        """
        guard = make_eq(
            make_func_apply(ident("hasToken"), ident("i")),
            bool_lit(True),
        )
        # If i = 0 and priorityPending: clear priority
        clear_prio = make_ite(
            make_land(
                make_eq(ident("i"), int_lit(0)),
                make_eq(ident("priorityPending"), bool_lit(True)),
            ),
            make_primed_eq("priorityPending", bool_lit(False)),
            make_unchanged("priorityPending"),
        )
        unchanged = make_conjunction([
            make_unchanged("token"),
            make_unchanged("hasToken"),
            make_unchanged("served"),
        ])
        return make_guard(guard, [clear_prio], ["token", "hasToken", "served"])

    def _build_next(self) -> Expression:
        """Next-state relation: disjunction of all actions."""
        actions: List[Expression] = []
        # PassToken(i) for each node
        actions.append(
            make_exists_single(
                "i", ident("Nodes"),
                _make_user_op_call("PassToken", ident("i")),
            )
        )
        # InitiatorRequest
        actions.append(ident("InitiatorRequest"))
        # ReceiveToken(i)
        actions.append(
            make_exists_single(
                "i", ident("Nodes"),
                _make_user_op_call("ReceiveToken", ident("i")),
            )
        )
        return make_disjunction(actions)

    # ------------------------------------------------------------------
    # Invariants and properties
    # ------------------------------------------------------------------

    def _build_mutex(self) -> Expression:
        """At most one node holds the token."""
        return make_forall_single(
            "i", ident("Nodes"),
            make_forall_single(
                "j", ident("Nodes"),
                make_implies(
                    make_land(
                        make_eq(
                            make_func_apply(ident("hasToken"), ident("i")),
                            bool_lit(True),
                        ),
                        make_eq(
                            make_func_apply(ident("hasToken"), ident("j")),
                            bool_lit(True),
                        ),
                    ),
                    make_eq(ident("i"), ident("j")),
                ),
            ),
        )

    def _build_all_served(self) -> Expression:
        """Liveness: eventually all nodes have been served."""
        return make_forall_single(
            "i", ident("Nodes"),
            make_eq(
                make_func_apply(ident("served"), ident("i")),
                bool_lit(True),
            ),
        )

    def _build_priority_served(self) -> Expression:
        """Priority liveness: if priority is pending, eventually served."""
        return make_implies(
            make_eq(ident("priorityPending"), bool_lit(True)),
            make_eq(ident("priorityPending"), bool_lit(False)),
        )

    def _build_fairness(self, vars_tuple: Expression) -> List[Expression]:
        """Build WF constraints for all actions."""
        wf_list: List[Expression] = []
        # WF for PassToken(i), for each node
        for i in range(self._n):
            pass_i = _make_user_op_call("PassToken", int_lit(i))
            wf_list.append(make_wf(vars_tuple, pass_i))
        # WF for InitiatorRequest
        wf_list.append(make_wf(vars_tuple, ident("InitiatorRequest")))
        return wf_list

    def _build_mutex_property(self) -> Property:
        return make_safety_property("Mutex", ident("Mutex"))

    def _build_liveness_property(self) -> Property:
        return make_liveness_property("AllServed", ident("AllServed"))

    def _build_priority_property(self) -> Property:
        return make_liveness_property("PriorityServed", ident("PriorityServed"))

    @staticmethod
    def _estimate_states(n: int) -> int:
        """Rough state-space estimate.

        State = (token ∈ {0..n-1}, hasToken ∈ {T,F}^n, priorityPending ∈ {T,F},
                 served ∈ {T,F}^n)
        Upper bound: n × 2^n × 2 × 2^n = n × 2^(2n+1)
        Reachable is much smaller due to mutex constraint.
        """
        return n * (2 ** (2 * n + 1))


def _make_user_op_call(name: str, *args: Expression) -> OperatorApplication:
    """Build a call to a user-defined operator."""
    return OperatorApplication(
        operator=Operator.FUNC_APPLY,
        operands=list(args),
        operator_name=name,
    )
