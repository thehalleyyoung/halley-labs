"""Ring-based Leader Election specification in TLA-lite AST form.

Models a simplified Chang-Roberts leader-election algorithm on a
unidirectional ring of N nodes.  Each node has a unique integer ID;
messages travel clockwise.  A node that receives an ID larger than its
own forwards it; upon receiving its own ID back it declares itself
leader.

Safety:   at most one node is elected leader.
Liveness: eventually exactly one leader is elected.
Agreement: the leader has the highest ID in the ring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..parser.ast_nodes import (
    AlwaysExpr,
    EventuallyExpr,
    Expression,
    Module,
    Operator,
    OperatorApplication,
    OperatorDef,
    Property,
    QuantifiedExpr,
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
    make_geq,
    make_gt,
    make_guard,
    make_implies,
    make_in,
    make_int_range,
    make_invariant_property,
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
    make_setdiff,
    make_spec_with_fairness,
    make_string_set,
    make_temporal_property,
    make_unchanged,
    make_union,
    make_vars_tuple,
    make_wf,
    primed,
    str_lit,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_VARIABLES = ("leader", "inbox", "active", "id")


class LeaderElectionSpec:
    """Programmatic builder for a ring-based Leader Election TLA-lite spec.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the ring (commonly 3 or 5).
    """

    def __init__(self, n_nodes: int = 3) -> None:
        if n_nodes < 2:
            raise ValueError("Need at least 2 nodes for leader election")
        self._n = n_nodes
        self._module: Optional[Module] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_spec(self) -> Module:
        """Return the TLA-lite Module AST for Leader Election."""
        if self._module is None:
            self._module = self._build_module()
        return self._module

    def get_properties(self) -> List[Property]:
        """Return the properties to model-check."""
        return [
            self._type_ok_property(),
            self._at_most_one_leader_property(),
            self._eventually_leader_property(),
            self._leader_has_max_id_property(),
            self._leader_stability_property(),
        ]

    def get_config(self, n_nodes: Optional[int] = None) -> Dict[str, Any]:
        n = n_nodes or self._n
        return {
            "spec_name": "LeaderElection",
            "n_nodes": n,
            "constants": {
                "Node": list(range(1, n + 1)),
                "Id": list(range(1, n + 1)),
            },
            "invariants": ["TypeOK", "AtMostOneLeader"],
            "properties": ["EventuallyLeader", "LeaderHasMaxId",
                           "LeaderStability"],
            "symmetry_sets": [],
            "state_constraint": None,
            "expected_states": self._estimate_states(n),
        }

    def validate(self) -> List[str]:
        """Validate the constructed specification."""
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
        required_defs = ("Init", "Next", "ProcessMessage",
                         "ReceiveMessage", "DeclareLeader",
                         "TypeOK", "AtMostOneLeader", "Spec")
        for rd in required_defs:
            if rd not in def_names:
                errors.append(f"Missing definition: {rd}")
        if not spec.properties:
            errors.append("No properties defined")
        return errors

    # ------------------------------------------------------------------
    # Module construction
    # ------------------------------------------------------------------

    def _build_module(self) -> Module:
        mb = ModuleBuilder("LeaderElection")
        mb.add_extends("Naturals", "FiniteSets")

        mb.add_constants("Node", "Id")
        mb.add_variables(*_ALL_VARIABLES)

        # Helper definitions
        mb.add_definition("Succ", self._build_successor(), params=["n"])
        mb.add_definition("MaxId", self._build_max_id())
        mb.add_definition("NodeIds", self._build_node_ids())

        # Type invariant
        mb.add_definition("TypeOK", self._build_type_ok())

        # Init
        mb.add_definition("Init", self._build_init())

        # Actions
        mb.add_definition("ProcessMessage",
                          self._build_process_message(), params=["n"])
        mb.add_definition("ReceiveMessage",
                          self._build_receive_message(), params=["n"])
        mb.add_definition("DeclareLeader",
                          self._build_declare_leader(), params=["n"])

        # Next
        mb.add_definition("Next", self._build_next())

        # Properties as definitions
        mb.add_definition("AtMostOneLeader",
                          self._build_at_most_one_leader())
        mb.add_definition("EventuallyLeader",
                          self._build_eventually_leader())
        mb.add_definition("LeaderHasMaxId",
                          self._build_leader_has_max_id())
        mb.add_definition("LeaderStability",
                          self._build_leader_stability())

        # Spec with fairness
        vars_t = make_vars_tuple(*_ALL_VARIABLES)
        fairness = self._build_fairness(vars_t)
        spec_expr = make_spec_with_fairness("Init", "Next", vars_t, fairness)
        mb.add_definition("Spec", spec_expr)

        # Register properties
        for prop in self.get_properties():
            mb.add_property(prop)

        return mb.build()

    # ------------------------------------------------------------------
    # Helper definitions
    # ------------------------------------------------------------------

    def _build_successor(self) -> Expression:
        """Succ(n) == IF n = N THEN 1 ELSE n + 1
        (clockwise neighbour in the ring)
        """
        from ..parser.ast_nodes import IfThenElse
        return IfThenElse(
            condition=make_eq(ident("n"), ident("N")),
            then_expr=int_lit(1),
            else_expr=make_plus(ident("n"), int_lit(1)),
        )

    def _build_max_id(self) -> Expression:
        """MaxId == CHOOSE m \\in Id : \\A i \\in Id : m >= i"""
        from ..parser.ast_nodes import ChooseExpr
        return ChooseExpr(
            variable="m",
            set_expr=ident("Id"),
            predicate=make_forall_single(
                "i", ident("Id"), make_geq(ident("m"), ident("i")),
            ),
        )

    def _build_node_ids(self) -> Expression:
        """NodeIds == [n \\in Node |-> n]  (each node's ID is its index)."""
        return make_function_construction("n", ident("Node"), ident("n"))

    # ------------------------------------------------------------------
    # Type invariant
    # ------------------------------------------------------------------

    def _build_type_ok(self) -> Expression:
        leader_ok = make_forall_single(
            "n", ident("Node"),
            make_in(
                make_func_apply(ident("leader"), ident("n")),
                make_set_enum(bool_lit(True), bool_lit(False)),
            ),
        )
        active_ok = make_forall_single(
            "n", ident("Node"),
            make_in(
                make_func_apply(ident("active"), ident("n")),
                make_set_enum(bool_lit(True), bool_lit(False)),
            ),
        )
        inbox_ok = make_forall_single(
            "n", ident("Node"),
            make_forall_single(
                "msg", make_func_apply(ident("inbox"), ident("n")),
                make_in(ident("msg"), ident("Id")),
            ),
        )
        id_ok = make_forall_single(
            "n", ident("Node"),
            make_in(make_func_apply(ident("id"), ident("n")), ident("Id")),
        )
        return make_conjunction([leader_ok, active_ok, inbox_ok, id_ok])

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _build_init(self) -> Expression:
        """Init ==
           /\\ leader = [n \\in Node |-> FALSE]
           /\\ active = [n \\in Node |-> TRUE]
           /\\ id     = NodeIds
           /\\ inbox  = [n \\in Node |-> {id[Succ(n)]}]
              (each node's clockwise neighbour sends its ID)
        """
        leader_init = make_eq(
            ident("leader"),
            make_function_construction("n", ident("Node"), bool_lit(False)),
        )
        active_init = make_eq(
            ident("active"),
            make_function_construction("n", ident("Node"), bool_lit(True)),
        )
        id_init = make_eq(ident("id"), ident("NodeIds"))
        # Initially each node sends its own id to its successor
        # inbox[succ(n)] gets n's id. For simplicity: inbox = [n \\in Node |-> {}]
        # then each node sends — but we model the initial send as part of Init:
        # inbox = [n \\in Node |-> {id[pred(n)]}]
        # Simpler: inbox starts empty; first action is SendId
        inbox_init = make_eq(
            ident("inbox"),
            make_function_construction("n", ident("Node"), make_set_enum()),
        )
        return make_conjunction([leader_init, active_init, id_init,
                                 inbox_init])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _build_process_message(self) -> Expression:
        """ProcessMessage(n) ==
           /\\ active[n] = TRUE
           /\\ inbox[n] # {}
           /\\ \\E msg \\in inbox[n] :
                /\\ inbox' = [inbox EXCEPT ![n] = inbox[n] \\ {msg}]
                /\\ IF msg > id[n]
                   THEN /\\ inbox' = [inbox EXCEPT
                              ![n] = inbox[n] \\ {msg},
                              ![Succ(n)] = inbox[Succ(n)] \\union {msg}]
                        /\\ UNCHANGED <<leader, active, id>>
                   ELSE IF msg = id[n]
                   THEN /\\ leader' = [leader EXCEPT ![n] = TRUE]
                        /\\ active' = [active EXCEPT ![n] = FALSE]
                        /\\ UNCHANGED <<id>>
                   ELSE /\\ UNCHANGED <<leader, active, id, inbox>>
                          \\ msg < id[n], discard
        """
        guard_active = make_eq(
            make_func_apply(ident("active"), ident("n")), bool_lit(True),
        )
        guard_nonempty = make_neq(
            make_func_apply(ident("inbox"), ident("n")), make_set_enum(),
        )

        msg_id = ident("msg")
        node_id = make_func_apply(ident("id"), ident("n"))
        succ_n = _make_user_call("Succ", ident("n"))

        # Case 1: msg > id[n] — forward msg to successor
        inbox_after_forward = make_except(
            ident("inbox"),
            [
                ([ident("n")],
                 make_setdiff(
                     make_func_apply(ident("inbox"), ident("n")),
                     make_set_enum(msg_id),
                 )),
                ([succ_n],
                 make_union(
                     make_func_apply(ident("inbox"), succ_n),
                     make_set_enum(msg_id),
                 )),
            ],
        )
        case_forward = make_conjunction([
            make_gt(msg_id, node_id),
            make_primed_eq("inbox", inbox_after_forward),
            make_unchanged("leader", "active", "id"),
        ])

        # Case 2: msg = id[n] — this node is the leader
        inbox_after_self = make_except(
            ident("inbox"),
            [([ident("n")],
              make_setdiff(
                  make_func_apply(ident("inbox"), ident("n")),
                  make_set_enum(msg_id),
              ))],
        )
        case_self = make_conjunction([
            make_eq(msg_id, node_id),
            make_primed_eq("inbox", inbox_after_self),
            make_primed_func_update("leader", ident("n"), bool_lit(True)),
            make_primed_func_update("active", ident("n"), bool_lit(False)),
            make_unchanged("id"),
        ])

        # Case 3: msg < id[n] — discard
        inbox_after_discard = make_except(
            ident("inbox"),
            [([ident("n")],
              make_setdiff(
                  make_func_apply(ident("inbox"), ident("n")),
                  make_set_enum(msg_id),
              ))],
        )
        case_discard = make_conjunction([
            make_lt_expr(msg_id, node_id),
            make_primed_eq("inbox", inbox_after_discard),
            make_unchanged("leader", "active", "id"),
        ])

        body = make_disjunction([case_forward, case_self, case_discard])
        exists_msg = make_exists_single(
            "msg", make_func_apply(ident("inbox"), ident("n")), body,
        )
        return make_conjunction([guard_active, guard_nonempty, exists_msg])

    def _build_receive_message(self) -> Expression:
        """ReceiveMessage(n) ==
           An active node sends its own id to its successor's inbox.
           /\\ active[n] = TRUE
           /\\ leader[n] = FALSE
           /\\ inbox' = [inbox EXCEPT ![Succ(n)] =
                          inbox[Succ(n)] \\union {id[n]}]
           /\\ UNCHANGED <<leader, active, id>>
        """
        succ_n = _make_user_call("Succ", ident("n"))
        guard = make_conjunction([
            make_eq(make_func_apply(ident("active"), ident("n")),
                    bool_lit(True)),
            make_eq(make_func_apply(ident("leader"), ident("n")),
                    bool_lit(False)),
        ])
        inbox_update = make_primed_eq(
            "inbox",
            make_except(
                ident("inbox"),
                [([succ_n],
                  make_union(
                      make_func_apply(ident("inbox"), succ_n),
                      make_set_enum(make_func_apply(ident("id"), ident("n"))),
                  ))],
            ),
        )
        return make_guard(guard, [inbox_update], ["leader", "active", "id"])

    def _build_declare_leader(self) -> Expression:
        """DeclareLeader(n) ==
           /\\ leader[n] = TRUE
           /\\ active[n] = FALSE
           /\\ UNCHANGED <<leader, active, inbox, id>>
           (stuttering step — leader already declared)
        """
        guard = make_conjunction([
            make_eq(make_func_apply(ident("leader"), ident("n")),
                    bool_lit(True)),
            make_eq(make_func_apply(ident("active"), ident("n")),
                    bool_lit(False)),
        ])
        return make_guard(guard, [], list(_ALL_VARIABLES))

    # ------------------------------------------------------------------
    # Next
    # ------------------------------------------------------------------

    def _build_next(self) -> Expression:
        """Next == \\E n \\in Node :
             \\/ ProcessMessage(n)
             \\/ ReceiveMessage(n)
             \\/ DeclareLeader(n)
        """
        per_node = make_disjunction([
            _make_user_call("ProcessMessage", ident("n")),
            _make_user_call("ReceiveMessage", ident("n")),
            _make_user_call("DeclareLeader", ident("n")),
        ])
        return make_exists_single("n", ident("Node"), per_node)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def _build_at_most_one_leader(self) -> Expression:
        """AtMostOneLeader ==
           \\A n1, n2 \\in Node :
             (leader[n1] = TRUE /\\ leader[n2] = TRUE) => n1 = n2
        """
        n1_leader = make_eq(
            make_func_apply(ident("leader"), ident("n1")), bool_lit(True),
        )
        n2_leader = make_eq(
            make_func_apply(ident("leader"), ident("n2")), bool_lit(True),
        )
        return QuantifiedExpr(
            quantifier="forall",
            variables=[("n1", ident("Node")), ("n2", ident("Node"))],
            body=make_implies(
                make_land(n1_leader, n2_leader),
                make_eq(ident("n1"), ident("n2")),
            ),
        )

    def _build_eventually_leader(self) -> Expression:
        """<>(\\E n \\in Node : leader[n] = TRUE)"""
        return EventuallyExpr(
            expr=make_exists_single(
                "n", ident("Node"),
                make_eq(
                    make_func_apply(ident("leader"), ident("n")),
                    bool_lit(True),
                ),
            ),
        )

    def _build_leader_has_max_id(self) -> Expression:
        """[](\\A n \\in Node : leader[n] = TRUE => id[n] = MaxId)"""
        return AlwaysExpr(
            expr=make_forall_single(
                "n", ident("Node"),
                make_implies(
                    make_eq(
                        make_func_apply(ident("leader"), ident("n")),
                        bool_lit(True),
                    ),
                    make_eq(
                        make_func_apply(ident("id"), ident("n")),
                        ident("MaxId"),
                    ),
                ),
            ),
        )

    def _build_leader_stability(self) -> Expression:
        """Once a leader is elected, it remains a leader.
        [](\\A n \\in Node : leader[n] = TRUE =>
             [](leader[n] = TRUE))
        """
        return AlwaysExpr(
            expr=make_forall_single(
                "n", ident("Node"),
                make_implies(
                    make_eq(
                        make_func_apply(ident("leader"), ident("n")),
                        bool_lit(True),
                    ),
                    AlwaysExpr(
                        expr=make_eq(
                            make_func_apply(ident("leader"), ident("n")),
                            bool_lit(True),
                        ),
                    ),
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Fairness
    # ------------------------------------------------------------------

    def _build_fairness(self, vars_tuple: Expression) -> List[Expression]:
        """WF for ProcessMessage and ReceiveMessage per node."""
        fairness_list: List[Expression] = []
        for action_name in ("ProcessMessage", "ReceiveMessage"):
            action_all = make_exists_single(
                "n", ident("Node"),
                _make_user_call(action_name, ident("n")),
            )
            fairness_list.append(make_wf(vars_tuple, action_all))
        return fairness_list

    # ------------------------------------------------------------------
    # Property wrappers
    # ------------------------------------------------------------------

    def _type_ok_property(self) -> Property:
        return make_invariant_property("TypeOK", ident("TypeOK"))

    def _at_most_one_leader_property(self) -> Property:
        return make_safety_property("AtMostOneLeader",
                                    ident("AtMostOneLeader"))

    def _eventually_leader_property(self) -> Property:
        return make_liveness_property("EventuallyLeader",
                                     ident("EventuallyLeader"))

    def _leader_has_max_id_property(self) -> Property:
        return make_safety_property("LeaderHasMaxId",
                                    ident("LeaderHasMaxId"))

    def _leader_stability_property(self) -> Property:
        return make_temporal_property("LeaderStability",
                                     ident("LeaderStability"))

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_states(n: int) -> Dict[str, int]:
        """Rough upper bound on state-space size."""
        leader_states = 2 ** n
        active_states = 2 ** n
        # inbox: each node can hold a subset of Id values
        inbox_states = (2 ** n) ** n
        total = leader_states * active_states * inbox_states
        return {
            "leader_states": leader_states,
            "active_states": active_states,
            "inbox_states_upper": inbox_states,
            "upper_bound": total,
            "note": "Reachable states much smaller due to ring constraints",
        }

    @staticmethod
    def supported_configurations() -> List[Dict[str, Any]]:
        return [
            {"name": "small", "n_nodes": 3,
             "description": "3-node ring election"},
            {"name": "medium", "n_nodes": 5,
             "description": "5-node ring election"},
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_user_call(name: str, *args: Expression) -> OperatorApplication:
    return OperatorApplication(
        operator=Operator.FUNC_APPLY,
        operands=list(args),
        operator_name=name,
    )


def make_lt_expr(a: Expression, b: Expression) -> OperatorApplication:
    return OperatorApplication(operator=Operator.LT, operands=[a, b])
