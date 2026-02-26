"""Two-Phase Commit protocol specification in TLA-lite AST form.

Builds a complete TLA-lite AST for the classic Two-Phase Commit protocol
parameterised by the number of resource managers (participants).

The protocol modeled:
  * A transaction manager (TM) coordinates N resource managers (RMs).
  * Each RM starts in "working" state and may prepare or abort.
  * The TM collects prepare messages; once all RMs are prepared it
    broadcasts commit; if any RM aborts it broadcasts abort.
  * Safety:  no RM is committed while another is aborted.
  * Liveness: under fairness all RMs eventually reach a terminal state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..parser.ast_nodes import (
    Expression,
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
    make_in,
    make_implies,
    make_int_range,
    make_invariant_property,
    make_land,
    make_liveness_property,
    make_lnot,
    make_lor,
    make_neq,
    make_notin,
    make_primed,
    make_primed_eq,
    make_primed_func_update,
    make_safety_property,
    make_set_enum,
    make_spec_with_fairness,
    make_string_set,
    make_subseteq,
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

_RM_STATES = ("working", "prepared", "committed", "aborted")
_TM_STATES = ("init", "committed", "aborted")

_ALL_VARIABLES = ("rmState", "tmState", "tmPrepared", "msgs")


class TwoPhaseCommitSpec:
    """Programmatic builder for a Two-Phase Commit TLA-lite specification.

    Parameters
    ----------
    n_participants : int
        Number of resource managers (RMs).  Commonly 3, 5, or 7.
    """

    def __init__(self, n_participants: int = 3) -> None:
        if n_participants < 2:
            raise ValueError("Need at least 2 participants for 2PC")
        self._n = n_participants
        self._module: Optional[Module] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_spec(self) -> Module:
        """Return the TLA-lite Module AST for Two-Phase Commit."""
        if self._module is None:
            self._module = self._build_module()
        return self._module

    def get_properties(self) -> List[Property]:
        """Return the list of properties to model-check."""
        return [
            self._build_type_ok_property(),
            self._build_consistency_property(),
            self._build_termination_property(),
            self._build_commit_validity_property(),
            self._build_abort_safety_property(),
        ]

    def get_config(self, n_participants: Optional[int] = None) -> Dict[str, Any]:
        """Return a configuration dict for model checking."""
        n = n_participants or self._n
        return {
            "spec_name": "TwoPhaseCommit",
            "n_participants": n,
            "constants": {"RM": list(range(1, n + 1))},
            "invariants": ["TypeOK", "Consistency"],
            "properties": ["Termination", "CommitValidity", "AbortSafety"],
            "symmetry_sets": ["RM"],
            "state_constraint": None,
            "expected_states": self._estimate_states(n),
        }

    def validate(self) -> List[str]:
        """Validate the constructed specification and return any errors."""
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
        for expected_def in ("Init", "Next", "TMRcvPrepared", "TMCommit",
                             "TMAbort", "RMPrepare", "RMChooseToAbort",
                             "RMRcvCommitMsg", "RMRcvAbortMsg",
                             "TypeOK", "Consistency", "Spec"):
            if expected_def not in def_names:
                errors.append(f"Missing definition: {expected_def}")
        if not spec.properties:
            errors.append("No properties defined")
        return errors

    # ------------------------------------------------------------------
    # Private — module construction
    # ------------------------------------------------------------------

    def _build_module(self) -> Module:
        """Assemble the full TwoPhaseCommit module."""
        mb = ModuleBuilder("TwoPhaseCommit")
        mb.add_extends("Naturals", "FiniteSets")

        mb.add_constants("RM")
        mb.add_variables(*_ALL_VARIABLES)

        # Helper sets
        mb.add_definition("RMStates", self._rm_states_set())
        mb.add_definition("TMStates", self._tm_states_set())
        mb.add_definition("Messages", self._messages_set())

        # Type invariant
        type_ok = self._build_type_ok()
        mb.add_definition("TypeOK", type_ok)

        # Init predicate
        mb.add_definition("Init", self._build_init())

        # Actions
        mb.add_definition("TMRcvPrepared", self._build_tm_rcv_prepared(),
                          params=["rm"])
        mb.add_definition("TMCommit", self._build_tm_commit())
        mb.add_definition("TMAbort", self._build_tm_abort())
        mb.add_definition("RMPrepare", self._build_rm_prepare(), params=["rm"])
        mb.add_definition("RMChooseToAbort", self._build_rm_choose_abort(),
                          params=["rm"])
        mb.add_definition("RMRcvCommitMsg", self._build_rm_rcv_commit(),
                          params=["rm"])
        mb.add_definition("RMRcvAbortMsg", self._build_rm_rcv_abort(),
                          params=["rm"])

        # Next-state relation
        mb.add_definition("Next", self._build_next())

        # Consistency invariant
        mb.add_definition("Consistency", self._build_consistency())

        # Fairness and Spec
        vars_t = make_vars_tuple(*_ALL_VARIABLES)
        fairness = self._build_fairness(vars_t)
        spec_expr = make_spec_with_fairness("Init", "Next", vars_t, fairness)
        mb.add_definition("Spec", spec_expr)

        # Termination property
        mb.add_definition("Termination", self._build_termination())

        # CommitValidity
        mb.add_definition("CommitValidity", self._build_commit_validity())

        # AbortSafety
        mb.add_definition("AbortSafety", self._build_abort_safety())

        # Properties
        for prop in self.get_properties():
            mb.add_property(prop)

        return mb.build()

    # ------------------------------------------------------------------
    # Helper-set definitions
    # ------------------------------------------------------------------

    def _rm_states_set(self) -> Expression:
        return make_string_set(*_RM_STATES)

    def _tm_states_set(self) -> Expression:
        return make_string_set(*_TM_STATES)

    def _messages_set(self) -> Expression:
        """Set of all possible message values.

        We model messages as strings: "Prepare", "Commit", "Abort",
        plus per-RM "Prepared_<rm>" messages.
        """
        return make_string_set("Commit", "Abort")

    # ------------------------------------------------------------------
    # Type invariant
    # ------------------------------------------------------------------

    def _build_type_ok(self) -> Expression:
        rm_state_ok = make_forall_single(
            "r", ident("RM"),
            make_in(
                make_func_apply(ident("rmState"), ident("r")),
                self._rm_states_set(),
            ),
        )
        tm_state_ok = make_in(ident("tmState"), self._tm_states_set())
        tm_prepared_ok = make_subseteq(ident("tmPrepared"), ident("RM"))
        msgs_ok = make_subseteq(
            ident("msgs"),
            make_union(
                make_string_set("Commit", "Abort"),
                make_set_enum(str_lit("Prepared")),
            ),
        )
        return make_conjunction([rm_state_ok, tm_state_ok,
                                 tm_prepared_ok, msgs_ok])

    # ------------------------------------------------------------------
    # Init predicate
    # ------------------------------------------------------------------

    def _build_init(self) -> Expression:
        rm_init = make_eq(
            ident("rmState"),
            make_function_construction("r", ident("RM"), str_lit("working")),
        )
        tm_init = make_eq(ident("tmState"), str_lit("init"))
        prepared_init = make_eq(ident("tmPrepared"), make_set_enum())
        msgs_init = make_eq(ident("msgs"), make_set_enum())
        return make_conjunction([rm_init, tm_init, prepared_init, msgs_init])

    # ------------------------------------------------------------------
    # TM actions
    # ------------------------------------------------------------------

    def _build_tm_rcv_prepared(self) -> Expression:
        """TMRcvPrepared(rm) ==
           /\\ tmState = "init"
           /\\ [type |-> "Prepared", rm |-> rm] \\in msgs
           /\\ tmPrepared' = tmPrepared \\union {rm}
           /\\ UNCHANGED <<rmState, tmState, msgs>>
        """
        guard = make_land(
            make_eq(ident("tmState"), str_lit("init")),
            make_in(str_lit("Prepared"), ident("msgs")),
        )
        update = make_primed_eq(
            "tmPrepared",
            make_union(ident("tmPrepared"), make_set_enum(ident("rm"))),
        )
        return make_guard(guard, [update], ["rmState", "tmState", "msgs"])

    def _build_tm_commit(self) -> Expression:
        """TMCommit ==
           /\\ tmState = "init"
           /\\ tmPrepared = RM
           /\\ tmState' = "committed"
           /\\ msgs' = msgs \\union {"Commit"}
           /\\ UNCHANGED <<rmState, tmPrepared>>
        """
        guard = make_land(
            make_eq(ident("tmState"), str_lit("init")),
            make_eq(ident("tmPrepared"), ident("RM")),
        )
        tm_update = make_primed_eq("tmState", str_lit("committed"))
        msgs_update = make_primed_eq(
            "msgs",
            make_union(ident("msgs"), make_set_enum(str_lit("Commit"))),
        )
        return make_guard(guard, [tm_update, msgs_update],
                          ["rmState", "tmPrepared"])

    def _build_tm_abort(self) -> Expression:
        """TMAbort ==
           /\\ tmState = "init"
           /\\ tmState' = "aborted"
           /\\ msgs' = msgs \\union {"Abort"}
           /\\ UNCHANGED <<rmState, tmPrepared>>
        """
        guard = make_eq(ident("tmState"), str_lit("init"))
        tm_update = make_primed_eq("tmState", str_lit("aborted"))
        msgs_update = make_primed_eq(
            "msgs",
            make_union(ident("msgs"), make_set_enum(str_lit("Abort"))),
        )
        return make_guard(guard, [tm_update, msgs_update],
                          ["rmState", "tmPrepared"])

    # ------------------------------------------------------------------
    # RM actions
    # ------------------------------------------------------------------

    def _build_rm_prepare(self) -> Expression:
        """RMPrepare(rm) ==
           /\\ rmState[rm] = "working"
           /\\ rmState' = [rmState EXCEPT ![rm] = "prepared"]
           /\\ msgs' = msgs \\union {"Prepared"}
           /\\ UNCHANGED <<tmState, tmPrepared>>
        """
        guard = make_eq(
            make_func_apply(ident("rmState"), ident("rm")),
            str_lit("working"),
        )
        rm_update = make_primed_func_update("rmState", ident("rm"),
                                            str_lit("prepared"))
        msgs_update = make_primed_eq(
            "msgs",
            make_union(ident("msgs"), make_set_enum(str_lit("Prepared"))),
        )
        return make_guard(guard, [rm_update, msgs_update],
                          ["tmState", "tmPrepared"])

    def _build_rm_choose_abort(self) -> Expression:
        """RMChooseToAbort(rm) ==
           /\\ rmState[rm] = "working"
           /\\ rmState' = [rmState EXCEPT ![rm] = "aborted"]
           /\\ UNCHANGED <<tmState, tmPrepared, msgs>>
        """
        guard = make_eq(
            make_func_apply(ident("rmState"), ident("rm")),
            str_lit("working"),
        )
        rm_update = make_primed_func_update("rmState", ident("rm"),
                                            str_lit("aborted"))
        return make_guard(guard, [rm_update],
                          ["tmState", "tmPrepared", "msgs"])

    def _build_rm_rcv_commit(self) -> Expression:
        """RMRcvCommitMsg(rm) ==
           /\\ "Commit" \\in msgs
           /\\ rmState' = [rmState EXCEPT ![rm] = "committed"]
           /\\ UNCHANGED <<tmState, tmPrepared, msgs>>
        """
        guard = make_in(str_lit("Commit"), ident("msgs"))
        rm_update = make_primed_func_update("rmState", ident("rm"),
                                            str_lit("committed"))
        return make_guard(guard, [rm_update],
                          ["tmState", "tmPrepared", "msgs"])

    def _build_rm_rcv_abort(self) -> Expression:
        """RMRcvAbortMsg(rm) ==
           /\\ "Abort" \\in msgs
           /\\ rmState' = [rmState EXCEPT ![rm] = "aborted"]
           /\\ UNCHANGED <<tmState, tmPrepared, msgs>>
        """
        guard = make_in(str_lit("Abort"), ident("msgs"))
        rm_update = make_primed_func_update("rmState", ident("rm"),
                                            str_lit("aborted"))
        return make_guard(guard, [rm_update],
                          ["tmState", "tmPrepared", "msgs"])

    # ------------------------------------------------------------------
    # Next-state relation
    # ------------------------------------------------------------------

    def _build_next(self) -> Expression:
        """Next ==
           \\/ TMCommit
           \\/ TMAbort
           \\/ \\E rm \\in RM :
                \\/ TMRcvPrepared(rm)
                \\/ RMPrepare(rm)
                \\/ RMChooseToAbort(rm)
                \\/ RMRcvCommitMsg(rm)
                \\/ RMRcvAbortMsg(rm)
        """
        per_rm_actions = make_disjunction([
            make_user_op_call("TMRcvPrepared", ident("rm")),
            make_user_op_call("RMPrepare", ident("rm")),
            make_user_op_call("RMChooseToAbort", ident("rm")),
            make_user_op_call("RMRcvCommitMsg", ident("rm")),
            make_user_op_call("RMRcvAbortMsg", ident("rm")),
        ])
        exists_rm = make_exists_single("rm", ident("RM"), per_rm_actions)
        return make_disjunction([
            ident("TMCommit"),
            ident("TMAbort"),
            exists_rm,
        ])

    # ------------------------------------------------------------------
    # Consistency invariant
    # ------------------------------------------------------------------

    def _build_consistency(self) -> Expression:
        """Consistency ==
           \\A r1, r2 \\in RM :
             ~ (rmState[r1] = "committed" /\\ rmState[r2] = "aborted")
        """
        r1_committed = make_eq(
            make_func_apply(ident("rmState"), ident("r1")),
            str_lit("committed"),
        )
        r2_aborted = make_eq(
            make_func_apply(ident("rmState"), ident("r2")),
            str_lit("aborted"),
        )
        from ..parser.ast_nodes import QuantifiedExpr
        return QuantifiedExpr(
            quantifier="forall",
            variables=[("r1", ident("RM")), ("r2", ident("RM"))],
            body=make_lnot(make_land(r1_committed, r2_aborted)),
        )

    # ------------------------------------------------------------------
    # Fairness
    # ------------------------------------------------------------------

    def _build_fairness(self, vars_tuple: Expression) -> List[Expression]:
        """WF for TMCommit, TMAbort, and for each RM action."""
        fairness_list: List[Expression] = [
            make_wf(vars_tuple, ident("TMCommit")),
            make_wf(vars_tuple, ident("TMAbort")),
        ]
        rm_actions = ["TMRcvPrepared", "RMPrepare", "RMChooseToAbort",
                      "RMRcvCommitMsg", "RMRcvAbortMsg"]
        for action_name in rm_actions:
            action_call = make_exists_single(
                "rm", ident("RM"),
                make_user_op_call(action_name, ident("rm")),
            )
            fairness_list.append(make_wf(vars_tuple, action_call))
        return fairness_list

    # ------------------------------------------------------------------
    # Temporal properties
    # ------------------------------------------------------------------

    def _build_termination(self) -> Expression:
        """Every RM eventually reaches "committed" or "aborted"."""
        from ..parser.ast_nodes import EventuallyExpr
        terminal = make_forall_single(
            "r", ident("RM"),
            make_lor(
                make_eq(make_func_apply(ident("rmState"), ident("r")),
                        str_lit("committed")),
                make_eq(make_func_apply(ident("rmState"), ident("r")),
                        str_lit("aborted")),
            ),
        )
        return EventuallyExpr(expr=terminal)

    def _build_commit_validity(self) -> Expression:
        """If all RMs prepare, eventually all commit.

        []( (\\A r \\in RM : rmState[r] \\in {"prepared","committed"})
             => <>(\\A r \\in RM : rmState[r] = "committed") )
        """
        from ..parser.ast_nodes import AlwaysExpr, EventuallyExpr
        all_prepared_or_committed = make_forall_single(
            "r", ident("RM"),
            make_in(
                make_func_apply(ident("rmState"), ident("r")),
                make_string_set("prepared", "committed"),
            ),
        )
        all_committed = make_forall_single(
            "r", ident("RM"),
            make_eq(
                make_func_apply(ident("rmState"), ident("r")),
                str_lit("committed"),
            ),
        )
        return AlwaysExpr(
            expr=make_implies(
                all_prepared_or_committed,
                EventuallyExpr(expr=all_committed),
            )
        )

    def _build_abort_safety(self) -> Expression:
        """If the TM aborts, no RM ever commits.

        [](tmState = "aborted" =>
           \\A r \\in RM : rmState[r] # "committed")
        """
        from ..parser.ast_nodes import AlwaysExpr
        tm_aborted = make_eq(ident("tmState"), str_lit("aborted"))
        no_rm_committed = make_forall_single(
            "r", ident("RM"),
            make_neq(
                make_func_apply(ident("rmState"), ident("r")),
                str_lit("committed"),
            ),
        )
        return AlwaysExpr(expr=make_implies(tm_aborted, no_rm_committed))

    # ------------------------------------------------------------------
    # Property wrappers
    # ------------------------------------------------------------------

    def _build_type_ok_property(self) -> Property:
        return make_invariant_property("TypeOK", ident("TypeOK"))

    def _build_consistency_property(self) -> Property:
        return make_safety_property("Consistency", ident("Consistency"))

    def _build_termination_property(self) -> Property:
        return make_liveness_property("Termination", ident("Termination"))

    def _build_commit_validity_property(self) -> Property:
        return make_temporal_property("CommitValidity",
                                      ident("CommitValidity"))

    def _build_abort_safety_property(self) -> Property:
        return make_safety_property("AbortSafety", ident("AbortSafety"))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_states(n: int) -> Dict[str, int]:
        """Rough estimate of the state-space size.

        rmState: 4^n  ×  tmState: 3  ×  tmPrepared: 2^n  ×  msgs: 2^3
        """
        rm_states = 4 ** n
        tm_states = 3
        tm_prepared = 2 ** n
        msgs = 8  # subsets of {Commit, Abort, Prepared}
        total = rm_states * tm_states * tm_prepared * msgs
        return {
            "rmState_states": rm_states,
            "tmState_states": tm_states,
            "tmPrepared_states": tm_prepared,
            "msgs_states": msgs,
            "upper_bound": total,
            "note": "Reachable states are much smaller due to protocol constraints",
        }

    @staticmethod
    def supported_configurations() -> List[Dict[str, Any]]:
        """Return preset configurations for benchmarking."""
        return [
            {"name": "small", "n_participants": 3,
             "description": "3-RM two-phase commit (fast)"},
            {"name": "medium", "n_participants": 5,
             "description": "5-RM two-phase commit (moderate)"},
            {"name": "large", "n_participants": 7,
             "description": "7-RM two-phase commit (large state space)"},
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def make_user_op_call(name: str, *args: Expression) -> OperatorApplication:
    """Build a call to a user-defined operator."""
    return OperatorApplication(
        operator=Operator.FUNC_APPLY,
        operands=list(args),
        operator_name=name,
    )
