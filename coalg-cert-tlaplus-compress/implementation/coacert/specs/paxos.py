"""Single-decree Paxos (Synod) specification in TLA-lite AST form.

Models the core Paxos consensus algorithm with N acceptors.  Proposers
and learners are implicit; the focus is on the message-passing protocol
between proposers and acceptors that guarantees agreement on a single
value.

Phases:
  1a  Proposer sends Prepare(bal)
  1b  Acceptor responds with Promise(bal, voted)
  2a  Proposer sends Accept(bal, val)  (after receiving a quorum of promises)
  2b  Acceptor votes Accept(bal, val)
  Decide  A learner observes a quorum of 2b messages for the same (bal,val)

Safety:   Agreement — at most one value is decided.
          Validity  — the decided value was proposed.
Liveness: Under fairness a decision is eventually reached.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..parser.ast_nodes import (
    AlwaysExpr,
    ChooseExpr,
    EventuallyExpr,
    Expression,
    Module,
    Operator,
    OperatorApplication,
    OperatorDef,
    Property,
    QuantifiedExpr,
    SetComprehension,
)
from .spec_utils import (
    ModuleBuilder,
    bool_lit,
    ident,
    int_lit,
    make_choose,
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
    make_lt,
    make_neq,
    make_notin,
    make_plus,
    make_primed_eq,
    make_primed_func_update,
    make_record,
    make_record_access,
    make_safety_property,
    make_set_comprehension,
    make_set_enum,
    make_set_map,
    make_spec_with_fairness,
    make_string_set,
    make_subseteq,
    make_temporal_property,
    make_unchanged,
    make_union,
    make_user_op,
    make_vars_tuple,
    make_wf,
    primed,
    str_lit,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MSG_TYPES = ("1a", "1b", "2a", "2b")
_ALL_VARIABLES = ("maxBal", "maxVBal", "maxVal", "msgs", "decided")


class PaxosSpec:
    """Programmatic builder for a single-decree Paxos TLA-lite spec.

    Parameters
    ----------
    n_acceptors : int
        Number of acceptors.  A quorum is any majority.
    n_values : int
        Number of distinct proposable values (default 2).
    max_ballot : int
        Maximum ballot number to bound the model (default 3).
    """

    def __init__(self, n_acceptors: int = 3, n_values: int = 2,
                 max_ballot: int = 3) -> None:
        if n_acceptors < 1:
            raise ValueError("Need at least 1 acceptor")
        if n_acceptors % 2 == 0:
            raise ValueError("Odd number of acceptors required for majority")
        self._n_acc = n_acceptors
        self._n_val = n_values
        self._max_bal = max_ballot
        self._module: Optional[Module] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_spec(self) -> Module:
        if self._module is None:
            self._module = self._build_module()
        return self._module

    def get_properties(self) -> List[Property]:
        return [
            self._type_ok_property(),
            self._agreement_property(),
            self._validity_property(),
            self._nontriviality_property(),
            self._eventual_decision_property(),
        ]

    def get_config(self, n_acceptors: Optional[int] = None) -> Dict[str, Any]:
        n = n_acceptors or self._n_acc
        quorum_size = n // 2 + 1
        return {
            "spec_name": "Paxos",
            "n_acceptors": n,
            "n_values": self._n_val,
            "max_ballot": self._max_bal,
            "quorum_size": quorum_size,
            "constants": {
                "Acceptor": list(range(1, n + 1)),
                "Value": [f"v{i}" for i in range(1, self._n_val + 1)],
                "Ballot": list(range(self._max_bal + 1)),
                "Quorum": self._enumerate_quorums(n),
            },
            "invariants": ["TypeOK", "Agreement", "Validity"],
            "properties": ["Nontriviality", "EventualDecision"],
            "symmetry_sets": ["Acceptor", "Value"],
            "state_constraint": f"maxBal values bounded by {self._max_bal}",
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
        required = ("Init", "Next", "Phase1a", "Phase1b",
                    "Phase2a", "Phase2b", "Decide",
                    "TypeOK", "Agreement", "Spec")
        for rd in required:
            if rd not in def_names:
                errors.append(f"Missing definition: {rd}")
        if not spec.properties:
            errors.append("No properties defined")
        return errors

    # ------------------------------------------------------------------
    # Module construction
    # ------------------------------------------------------------------

    def _build_module(self) -> Module:
        mb = ModuleBuilder("Paxos")
        mb.add_extends("Naturals", "FiniteSets")

        mb.add_constants("Acceptor", "Value", "Ballot", "Quorum")
        mb.add_variables(*_ALL_VARIABLES)

        # Helper operators
        mb.add_definition("None_", str_lit("None"))
        mb.add_definition("MaxBallot", int_lit(self._max_bal))
        mb.add_definition("QuorumAssumption", self._build_quorum_assumption())
        mb.add_definition("IsQuorum", self._build_is_quorum(), params=["Q"])

        mb.add_definition("Msgs1a", self._build_msgs_1a_set())
        mb.add_definition("Msgs1b", self._build_msgs_1b_set())
        mb.add_definition("Msgs2a", self._build_msgs_2a_set())
        mb.add_definition("Msgs2b", self._build_msgs_2b_set())
        mb.add_definition("AllMessages", self._build_all_messages_set())

        # Type invariant
        mb.add_definition("TypeOK", self._build_type_ok())

        # Init
        mb.add_definition("Init", self._build_init())

        # Helper: ShowsSafeAt
        mb.add_definition("ShowsSafeAt",
                          self._build_shows_safe_at(),
                          params=["Q", "b", "v"])

        # Actions
        mb.add_definition("Phase1a", self._build_phase_1a(), params=["b"])
        mb.add_definition("Phase1b", self._build_phase_1b(),
                          params=["a", "b"])
        mb.add_definition("Phase2a", self._build_phase_2a(),
                          params=["b", "v"])
        mb.add_definition("Phase2b", self._build_phase_2b(),
                          params=["a", "b", "v"])
        mb.add_definition("Decide", self._build_decide(),
                          params=["b", "v"])

        # Next
        mb.add_definition("Next", self._build_next())

        # Properties as definitions
        mb.add_definition("Agreement", self._build_agreement())
        mb.add_definition("Validity", self._build_validity())
        mb.add_definition("Nontriviality", self._build_nontriviality())
        mb.add_definition("EventualDecision",
                          self._build_eventual_decision())

        # Spec
        vars_t = make_vars_tuple(*_ALL_VARIABLES)
        fairness = self._build_fairness(vars_t)
        spec_expr = make_spec_with_fairness("Init", "Next", vars_t, fairness)
        mb.add_definition("Spec", spec_expr)

        for prop in self.get_properties():
            mb.add_property(prop)
        return mb.build()

    # ------------------------------------------------------------------
    # Helper set definitions
    # ------------------------------------------------------------------

    def _build_quorum_assumption(self) -> Expression:
        """\\A Q1, Q2 \\in Quorum : Q1 \\cap Q2 # {}"""
        return QuantifiedExpr(
            quantifier="forall",
            variables=[("Q1", ident("Quorum")), ("Q2", ident("Quorum"))],
            body=make_neq(
                OperatorApplication(operator=Operator.INTERSECT,
                                    operands=[ident("Q1"), ident("Q2")]),
                make_set_enum(),
            ),
        )

    def _build_is_quorum(self) -> Expression:
        """IsQuorum(Q) == Q \\in Quorum"""
        return make_in(ident("Q"), ident("Quorum"))

    def _build_msgs_1a_set(self) -> Expression:
        """Set of 1a message records: [type |-> "1a", bal |-> b]"""
        return make_set_map(
            "b", ident("Ballot"),
            make_record(("type", str_lit("1a")), ("bal", ident("b"))),
        )

    def _build_msgs_1b_set(self) -> Expression:
        """Set of 1b messages: [type:"1b", acc:a, bal:b, mbal:mb, mval:mv]"""
        # Simplified: just describe the shape
        return make_set_map(
            "a", ident("Acceptor"),
            make_record(
                ("type", str_lit("1b")),
                ("acc", ident("a")),
                ("bal", int_lit(0)),
                ("mbal", int_lit(-1)),
                ("mval", ident("None_")),
            ),
        )

    def _build_msgs_2a_set(self) -> Expression:
        return make_set_map(
            "b", ident("Ballot"),
            make_record(
                ("type", str_lit("2a")),
                ("bal", ident("b")),
                ("val", str_lit("v")),
            ),
        )

    def _build_msgs_2b_set(self) -> Expression:
        return make_set_map(
            "a", ident("Acceptor"),
            make_record(
                ("type", str_lit("2b")),
                ("acc", ident("a")),
                ("bal", int_lit(0)),
                ("val", str_lit("v")),
            ),
        )

    def _build_all_messages_set(self) -> Expression:
        """Union of all message type sets."""
        return make_union(
            make_union(ident("Msgs1a"), ident("Msgs1b")),
            make_union(ident("Msgs2a"), ident("Msgs2b")),
        )

    # ------------------------------------------------------------------
    # Type invariant
    # ------------------------------------------------------------------

    def _build_type_ok(self) -> Expression:
        maxbal_ok = make_forall_single(
            "a", ident("Acceptor"),
            make_in(
                make_func_apply(ident("maxBal"), ident("a")),
                make_union(
                    ident("Ballot"),
                    make_set_enum(int_lit(-1)),
                ),
            ),
        )
        maxvbal_ok = make_forall_single(
            "a", ident("Acceptor"),
            make_in(
                make_func_apply(ident("maxVBal"), ident("a")),
                make_union(
                    ident("Ballot"),
                    make_set_enum(int_lit(-1)),
                ),
            ),
        )
        maxval_ok = make_forall_single(
            "a", ident("Acceptor"),
            make_lor(
                make_in(
                    make_func_apply(ident("maxVal"), ident("a")),
                    ident("Value"),
                ),
                make_eq(
                    make_func_apply(ident("maxVal"), ident("a")),
                    ident("None_"),
                ),
            ),
        )
        decided_ok = make_subseteq(ident("decided"), ident("Value"))
        return make_conjunction([maxbal_ok, maxvbal_ok, maxval_ok,
                                 decided_ok])

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _build_init(self) -> Expression:
        maxbal_init = make_eq(
            ident("maxBal"),
            make_function_construction("a", ident("Acceptor"), int_lit(-1)),
        )
        maxvbal_init = make_eq(
            ident("maxVBal"),
            make_function_construction("a", ident("Acceptor"), int_lit(-1)),
        )
        maxval_init = make_eq(
            ident("maxVal"),
            make_function_construction("a", ident("Acceptor"),
                                       ident("None_")),
        )
        msgs_init = make_eq(ident("msgs"), make_set_enum())
        decided_init = make_eq(ident("decided"), make_set_enum())
        return make_conjunction([maxbal_init, maxvbal_init, maxval_init,
                                 msgs_init, decided_init])

    # ------------------------------------------------------------------
    # ShowsSafeAt helper
    # ------------------------------------------------------------------

    def _build_shows_safe_at(self) -> Expression:
        """ShowsSafeAt(Q, b, v) ==
           /\\ \\A a \\in Q :
                \\E m \\in msgs :
                  /\\ m.type = "1b"
                  /\\ m.acc = a
                  /\\ m.bal = b
           /\\ \\/ \\A a \\in Q :
                   \\E m \\in msgs :
                     /\\ m.type = "1b" /\\ m.acc = a /\\ m.bal = b
                     /\\ m.mbal = -1
              \\/ \\E c \\in Ballot :
                   /\\ c < b
                   /\\ (\\E a \\in Q :
                         \\E m \\in msgs :
                           m.type = "1b" /\\ m.acc = a /\\ m.bal = b
                           /\\ m.mbal = c /\\ m.mval = v)
                   /\\ \\A a \\in Q :
                       \\E m \\in msgs :
                         m.type = "1b" /\\ m.acc = a /\\ m.bal = b
                         /\\ m.mbal =< c
        """
        # All Q members have sent 1b for ballot b
        has_1b = make_forall_single(
            "a", ident("Q"),
            make_exists_single(
                "m", ident("msgs"),
                make_conjunction([
                    make_eq(make_record_access(ident("m"), "type"),
                            str_lit("1b")),
                    make_eq(make_record_access(ident("m"), "acc"),
                            ident("a")),
                    make_eq(make_record_access(ident("m"), "bal"),
                            ident("b")),
                ]),
            ),
        )

        # Case 1: all 1b messages have mbal = -1 (no prior vote)
        all_none = make_forall_single(
            "a", ident("Q"),
            make_exists_single(
                "m", ident("msgs"),
                make_conjunction([
                    make_eq(make_record_access(ident("m"), "type"),
                            str_lit("1b")),
                    make_eq(make_record_access(ident("m"), "acc"),
                            ident("a")),
                    make_eq(make_record_access(ident("m"), "bal"),
                            ident("b")),
                    make_eq(make_record_access(ident("m"), "mbal"),
                            int_lit(-1)),
                ]),
            ),
        )

        # Case 2: some acceptor voted at ballot c < b with value v
        witness = make_exists_single(
            "a2", ident("Q"),
            make_exists_single(
                "m2", ident("msgs"),
                make_conjunction([
                    make_eq(make_record_access(ident("m2"), "type"),
                            str_lit("1b")),
                    make_eq(make_record_access(ident("m2"), "acc"),
                            ident("a2")),
                    make_eq(make_record_access(ident("m2"), "bal"),
                            ident("b")),
                    make_eq(make_record_access(ident("m2"), "mbal"),
                            ident("c")),
                    make_eq(make_record_access(ident("m2"), "mval"),
                            ident("v")),
                ]),
            ),
        )
        all_leq = make_forall_single(
            "a3", ident("Q"),
            make_exists_single(
                "m3", ident("msgs"),
                make_conjunction([
                    make_eq(make_record_access(ident("m3"), "type"),
                            str_lit("1b")),
                    make_eq(make_record_access(ident("m3"), "acc"),
                            ident("a3")),
                    make_eq(make_record_access(ident("m3"), "bal"),
                            ident("b")),
                    OperatorApplication(
                        operator=Operator.LEQ,
                        operands=[
                            make_record_access(ident("m3"), "mbal"),
                            ident("c"),
                        ],
                    ),
                ]),
            ),
        )
        case2 = make_exists_single(
            "c", ident("Ballot"),
            make_conjunction([
                make_lt(ident("c"), ident("b")),
                witness,
                all_leq,
            ]),
        )

        return make_conjunction([has_1b, make_lor(all_none, case2)])

    # ------------------------------------------------------------------
    # Phase 1a: Proposer sends Prepare
    # ------------------------------------------------------------------

    def _build_phase_1a(self) -> Expression:
        """Phase1a(b) ==
           /\\ ~ \\E m \\in msgs : m.type = "1a" /\\ m.bal = b
           /\\ msgs' = msgs \\union {[type |-> "1a", bal |-> b]}
           /\\ UNCHANGED <<maxBal, maxVBal, maxVal, decided>>
        """
        no_existing = make_lnot(
            make_exists_single(
                "m", ident("msgs"),
                make_land(
                    make_eq(make_record_access(ident("m"), "type"),
                            str_lit("1a")),
                    make_eq(make_record_access(ident("m"), "bal"),
                            ident("b")),
                ),
            ),
        )
        new_msg = make_record(("type", str_lit("1a")), ("bal", ident("b")))
        msgs_update = make_primed_eq(
            "msgs", make_union(ident("msgs"), make_set_enum(new_msg)),
        )
        return make_guard(no_existing, [msgs_update],
                          ["maxBal", "maxVBal", "maxVal", "decided"])

    # ------------------------------------------------------------------
    # Phase 1b: Acceptor sends Promise
    # ------------------------------------------------------------------

    def _build_phase_1b(self) -> Expression:
        """Phase1b(a, b) ==
           /\\ b > maxBal[a]
           /\\ \\E m \\in msgs : m.type = "1a" /\\ m.bal = b
           /\\ maxBal' = [maxBal EXCEPT ![a] = b]
           /\\ msgs' = msgs \\union
                {[type |-> "1b", acc |-> a, bal |-> b,
                  mbal |-> maxVBal[a], mval |-> maxVal[a]]}
           /\\ UNCHANGED <<maxVBal, maxVal, decided>>
        """
        guard_bal = make_gt(ident("b"),
                            make_func_apply(ident("maxBal"), ident("a")))
        guard_1a = make_exists_single(
            "m", ident("msgs"),
            make_land(
                make_eq(make_record_access(ident("m"), "type"),
                        str_lit("1a")),
                make_eq(make_record_access(ident("m"), "bal"), ident("b")),
            ),
        )
        guard = make_land(guard_bal, guard_1a)
        maxbal_update = make_primed_func_update("maxBal", ident("a"),
                                                ident("b"))
        new_msg = make_record(
            ("type", str_lit("1b")),
            ("acc", ident("a")),
            ("bal", ident("b")),
            ("mbal", make_func_apply(ident("maxVBal"), ident("a"))),
            ("mval", make_func_apply(ident("maxVal"), ident("a"))),
        )
        msgs_update = make_primed_eq(
            "msgs", make_union(ident("msgs"), make_set_enum(new_msg)),
        )
        return make_guard(guard, [maxbal_update, msgs_update],
                          ["maxVBal", "maxVal", "decided"])

    # ------------------------------------------------------------------
    # Phase 2a: Proposer sends Accept
    # ------------------------------------------------------------------

    def _build_phase_2a(self) -> Expression:
        """Phase2a(b, v) ==
           /\\ ~ \\E m \\in msgs : m.type = "2a" /\\ m.bal = b
           /\\ \\E Q \\in Quorum : ShowsSafeAt(Q, b, v)
           /\\ msgs' = msgs \\union
                {[type |-> "2a", bal |-> b, val |-> v]}
           /\\ UNCHANGED <<maxBal, maxVBal, maxVal, decided>>
        """
        no_existing_2a = make_lnot(
            make_exists_single(
                "m", ident("msgs"),
                make_land(
                    make_eq(make_record_access(ident("m"), "type"),
                            str_lit("2a")),
                    make_eq(make_record_access(ident("m"), "bal"),
                            ident("b")),
                ),
            ),
        )
        quorum_safe = make_exists_single(
            "Q", ident("Quorum"),
            _make_user_call("ShowsSafeAt", ident("Q"), ident("b"),
                            ident("v")),
        )
        guard = make_land(no_existing_2a, quorum_safe)
        new_msg = make_record(
            ("type", str_lit("2a")),
            ("bal", ident("b")),
            ("val", ident("v")),
        )
        msgs_update = make_primed_eq(
            "msgs", make_union(ident("msgs"), make_set_enum(new_msg)),
        )
        return make_guard(guard, [msgs_update],
                          ["maxBal", "maxVBal", "maxVal", "decided"])

    # ------------------------------------------------------------------
    # Phase 2b: Acceptor votes
    # ------------------------------------------------------------------

    def _build_phase_2b(self) -> Expression:
        """Phase2b(a, b, v) ==
           /\\ \\E m \\in msgs : m.type = "2a" /\\ m.bal = b /\\ m.val = v
           /\\ b >= maxBal[a]
           /\\ maxBal'  = [maxBal  EXCEPT ![a] = b]
           /\\ maxVBal' = [maxVBal EXCEPT ![a] = b]
           /\\ maxVal'  = [maxVal  EXCEPT ![a] = v]
           /\\ msgs' = msgs \\union
                {[type |-> "2b", acc |-> a, bal |-> b, val |-> v]}
           /\\ UNCHANGED decided
        """
        guard_2a = make_exists_single(
            "m", ident("msgs"),
            make_conjunction([
                make_eq(make_record_access(ident("m"), "type"),
                        str_lit("2a")),
                make_eq(make_record_access(ident("m"), "bal"), ident("b")),
                make_eq(make_record_access(ident("m"), "val"), ident("v")),
            ]),
        )
        guard_bal = make_geq(ident("b"),
                             make_func_apply(ident("maxBal"), ident("a")))
        guard = make_land(guard_2a, guard_bal)

        maxbal_upd = make_primed_func_update("maxBal", ident("a"),
                                             ident("b"))
        maxvbal_upd = make_primed_func_update("maxVBal", ident("a"),
                                              ident("b"))
        maxval_upd = make_primed_func_update("maxVal", ident("a"),
                                             ident("v"))
        new_msg = make_record(
            ("type", str_lit("2b")),
            ("acc", ident("a")),
            ("bal", ident("b")),
            ("val", ident("v")),
        )
        msgs_upd = make_primed_eq(
            "msgs", make_union(ident("msgs"), make_set_enum(new_msg)),
        )
        return make_guard(guard,
                          [maxbal_upd, maxvbal_upd, maxval_upd, msgs_upd],
                          ["decided"])

    # ------------------------------------------------------------------
    # Decide: learner sees a quorum of 2b's
    # ------------------------------------------------------------------

    def _build_decide(self) -> Expression:
        """Decide(b, v) ==
           /\\ \\E Q \\in Quorum :
                \\A a \\in Q :
                  \\E m \\in msgs :
                    m.type = "2b" /\\ m.acc = a
                    /\\ m.bal = b /\\ m.val = v
           /\\ decided' = decided \\union {v}
           /\\ UNCHANGED <<maxBal, maxVBal, maxVal, msgs>>
        """
        quorum_voted = make_exists_single(
            "Q", ident("Quorum"),
            make_forall_single(
                "a", ident("Q"),
                make_exists_single(
                    "m", ident("msgs"),
                    make_conjunction([
                        make_eq(make_record_access(ident("m"), "type"),
                                str_lit("2b")),
                        make_eq(make_record_access(ident("m"), "acc"),
                                ident("a")),
                        make_eq(make_record_access(ident("m"), "bal"),
                                ident("b")),
                        make_eq(make_record_access(ident("m"), "val"),
                                ident("v")),
                    ]),
                ),
            ),
        )
        decided_upd = make_primed_eq(
            "decided",
            make_union(ident("decided"), make_set_enum(ident("v"))),
        )
        return make_guard(quorum_voted, [decided_upd],
                          ["maxBal", "maxVBal", "maxVal", "msgs"])

    # ------------------------------------------------------------------
    # Next-state relation
    # ------------------------------------------------------------------

    def _build_next(self) -> Expression:
        """Next ==
           \\/ \\E b \\in Ballot : Phase1a(b)
           \\/ \\E a \\in Acceptor, b \\in Ballot : Phase1b(a, b)
           \\/ \\E b \\in Ballot, v \\in Value : Phase2a(b, v)
           \\/ \\E a \\in Acceptor, b \\in Ballot, v \\in Value :
                Phase2b(a, b, v)
           \\/ \\E b \\in Ballot, v \\in Value : Decide(b, v)
        """
        phase_1a = make_exists_single(
            "b", ident("Ballot"),
            _make_user_call("Phase1a", ident("b")),
        )
        phase_1b = QuantifiedExpr(
            quantifier="exists",
            variables=[("a", ident("Acceptor")), ("b", ident("Ballot"))],
            body=_make_user_call("Phase1b", ident("a"), ident("b")),
        )
        phase_2a = QuantifiedExpr(
            quantifier="exists",
            variables=[("b", ident("Ballot")), ("v", ident("Value"))],
            body=_make_user_call("Phase2a", ident("b"), ident("v")),
        )
        phase_2b = QuantifiedExpr(
            quantifier="exists",
            variables=[("a", ident("Acceptor")),
                       ("b", ident("Ballot")),
                       ("v", ident("Value"))],
            body=_make_user_call("Phase2b", ident("a"), ident("b"),
                                 ident("v")),
        )
        decide = QuantifiedExpr(
            quantifier="exists",
            variables=[("b", ident("Ballot")), ("v", ident("Value"))],
            body=_make_user_call("Decide", ident("b"), ident("v")),
        )
        return make_disjunction([phase_1a, phase_1b, phase_2a,
                                  phase_2b, decide])

    # ------------------------------------------------------------------
    # Safety properties
    # ------------------------------------------------------------------

    def _build_agreement(self) -> Expression:
        """Agreement ==
           \\A v1, v2 \\in decided : v1 = v2
        """
        return QuantifiedExpr(
            quantifier="forall",
            variables=[("v1", ident("decided")),
                       ("v2", ident("decided"))],
            body=make_eq(ident("v1"), ident("v2")),
        )

    def _build_validity(self) -> Expression:
        """Validity == decided \\subseteq Value"""
        return make_subseteq(ident("decided"), ident("Value"))

    def _build_nontriviality(self) -> Expression:
        """Nontriviality ==
           \\A v \\in decided :
             \\E m \\in msgs : m.type = "2a" /\\ m.val = v
        (decided value was actually proposed in a 2a message)
        """
        return make_forall_single(
            "v", ident("decided"),
            make_exists_single(
                "m", ident("msgs"),
                make_land(
                    make_eq(make_record_access(ident("m"), "type"),
                            str_lit("2a")),
                    make_eq(make_record_access(ident("m"), "val"),
                            ident("v")),
                ),
            ),
        )

    def _build_eventual_decision(self) -> Expression:
        """EventualDecision == <>(decided # {})"""
        return EventuallyExpr(
            expr=make_neq(ident("decided"), make_set_enum()),
        )

    # ------------------------------------------------------------------
    # Fairness
    # ------------------------------------------------------------------

    def _build_fairness(self, vars_tuple: Expression) -> List[Expression]:
        fairness_list: List[Expression] = []

        # WF for Phase1a
        p1a_all = make_exists_single(
            "b", ident("Ballot"),
            _make_user_call("Phase1a", ident("b")),
        )
        fairness_list.append(make_wf(vars_tuple, p1a_all))

        # WF for Phase1b
        p1b_all = QuantifiedExpr(
            quantifier="exists",
            variables=[("a", ident("Acceptor")), ("b", ident("Ballot"))],
            body=_make_user_call("Phase1b", ident("a"), ident("b")),
        )
        fairness_list.append(make_wf(vars_tuple, p1b_all))

        # WF for Phase2a
        p2a_all = QuantifiedExpr(
            quantifier="exists",
            variables=[("b", ident("Ballot")), ("v", ident("Value"))],
            body=_make_user_call("Phase2a", ident("b"), ident("v")),
        )
        fairness_list.append(make_wf(vars_tuple, p2a_all))

        # WF for Phase2b
        p2b_all = QuantifiedExpr(
            quantifier="exists",
            variables=[("a", ident("Acceptor")),
                       ("b", ident("Ballot")),
                       ("v", ident("Value"))],
            body=_make_user_call("Phase2b", ident("a"), ident("b"),
                                 ident("v")),
        )
        fairness_list.append(make_wf(vars_tuple, p2b_all))

        # WF for Decide
        decide_all = QuantifiedExpr(
            quantifier="exists",
            variables=[("b", ident("Ballot")), ("v", ident("Value"))],
            body=_make_user_call("Decide", ident("b"), ident("v")),
        )
        fairness_list.append(make_wf(vars_tuple, decide_all))

        return fairness_list

    # ------------------------------------------------------------------
    # Property wrappers
    # ------------------------------------------------------------------

    def _type_ok_property(self) -> Property:
        return make_invariant_property("TypeOK", ident("TypeOK"))

    def _agreement_property(self) -> Property:
        return make_safety_property("Agreement", ident("Agreement"))

    def _validity_property(self) -> Property:
        return make_invariant_property("Validity", ident("Validity"))

    def _nontriviality_property(self) -> Property:
        return make_invariant_property("Nontriviality",
                                      ident("Nontriviality"))

    def _eventual_decision_property(self) -> Property:
        return make_liveness_property("EventualDecision",
                                     ident("EventualDecision"))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _enumerate_quorums(n: int) -> List[List[int]]:
        """Enumerate all majority subsets of {1..n}."""
        from itertools import combinations
        majority = n // 2 + 1
        acceptors = list(range(1, n + 1))
        quorums = []
        for size in range(majority, n + 1):
            for combo in combinations(acceptors, size):
                quorums.append(list(combo))
        return quorums

    @staticmethod
    def _estimate_states(n: int) -> Dict[str, int]:
        bal_range = 5  # ballots 0..4 by default
        val_range = 2
        # Per acceptor: maxBal * maxVBal * maxVal
        per_acc = (bal_range + 1) * (bal_range + 1) * (val_range + 1)
        acc_states = per_acc ** n
        # msgs is a set — exponential in message types
        msg_types = bal_range * 4  # rough
        msgs_states = 2 ** min(msg_types, 20)
        decided_states = 2 ** val_range
        total = acc_states * decided_states  # msgs not counted fully
        return {
            "per_acceptor_states": per_acc,
            "acceptor_states": acc_states,
            "decided_states": decided_states,
            "upper_bound": total,
            "note": ("Actual reachable states far smaller; msgs set "
                     "grows monotonically"),
        }

    @staticmethod
    def supported_configurations() -> List[Dict[str, Any]]:
        return [
            {"name": "small", "n_acceptors": 3, "n_values": 2,
             "max_ballot": 2,
             "description": "3-acceptor Paxos with 2 values, 3 ballots"},
            {"name": "medium", "n_acceptors": 3, "n_values": 2,
             "max_ballot": 3,
             "description": "3-acceptor Paxos with 2 values, 4 ballots"},
            {"name": "large", "n_acceptors": 5, "n_values": 2,
             "max_ballot": 3,
             "description": "5-acceptor Paxos (large state space)"},
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
