"""Peterson's Mutual Exclusion specification in TLA-lite AST form.

For N=2 this is the classic Peterson's algorithm.  For N>=3 the
specification generalises to the *filter lock* (a.k.a. generalised
Peterson's algorithm) which uses N-1 levels of competition.

Safety:     mutual exclusion (at most one process in the critical section).
Liveness:   starvation freedom — every process that wants to enter the
            critical section eventually does.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..parser.ast_nodes import (
    AlwaysExpr,
    EventuallyExpr,
    Expression,
    IfThenElse,
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
    make_leads_to,
    make_leq,
    make_liveness_property,
    make_lnot,
    make_lor,
    make_lt,
    make_minus,
    make_neq,
    make_plus,
    make_primed_eq,
    make_primed_func_update,
    make_safety_property,
    make_set_enum,
    make_spec_with_fairness,
    make_string_set,
    make_temporal_property,
    make_unchanged,
    make_vars_tuple,
    make_wf,
    primed,
    str_lit,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PC_VALUES = ("idle", "want", "wait", "critical")
_ALL_VARS_2 = ("pc", "flag", "turn")
_ALL_VARS_N = ("pc", "level", "last_to_enter", "waiting")


class PetersonSpec:
    """Programmatic builder for Peterson's Mutual Exclusion TLA-lite spec.

    Parameters
    ----------
    n_processes : int
        Number of competing processes (2 or 3).
        N=2 uses classic Peterson's; N>=3 uses the filter lock.
    """

    def __init__(self, n_processes: int = 2) -> None:
        if n_processes < 2:
            raise ValueError("Need at least 2 processes")
        self._n = n_processes
        self._module: Optional[Module] = None

    @property
    def _is_classic(self) -> bool:
        return self._n == 2

    @property
    def _var_names(self) -> tuple:
        return _ALL_VARS_2 if self._is_classic else _ALL_VARS_N

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
            self._mutex_property(),
            self._starvation_freedom_property(),
            self._eventually_critical_property(),
        ]

    def get_config(self, n_processes: Optional[int] = None) -> Dict[str, Any]:
        n = n_processes or self._n
        return {
            "spec_name": "Peterson",
            "n_processes": n,
            "algorithm": "classic" if n == 2 else "filter_lock",
            "constants": {"Proc": list(range(n))},
            "invariants": ["TypeOK", "MutualExclusion"],
            "properties": ["StarvationFreedom", "EventuallyCritical"],
            "symmetry_sets": ["Proc"],
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
        for expected in self._var_names:
            if expected not in var_names:
                errors.append(f"Missing variable: {expected}")
        def_names = {d.name for d in spec.definitions
                     if isinstance(d, OperatorDef)}
        for rd in ("Init", "Next", "TypeOK", "MutualExclusion", "Spec"):
            if rd not in def_names:
                errors.append(f"Missing definition: {rd}")
        if not spec.properties:
            errors.append("No properties defined")
        return errors

    # ------------------------------------------------------------------
    # Module construction — dispatch
    # ------------------------------------------------------------------

    def _build_module(self) -> Module:
        if self._is_classic:
            return self._build_classic_module()
        return self._build_filter_lock_module()

    # ==================================================================
    # Classic Peterson's (N=2)
    # ==================================================================

    def _build_classic_module(self) -> Module:
        mb = ModuleBuilder("Peterson")
        mb.add_extends("Naturals")
        mb.add_constants("Proc")
        mb.add_variables(*_ALL_VARS_2)

        # Proc set: {0, 1}
        proc_set = ident("Proc")
        other_proc = self._classic_other()

        mb.add_definition("Other", other_proc, params=["p"])
        mb.add_definition("PCValues", make_string_set(*_PC_VALUES))
        mb.add_definition("TypeOK", self._classic_type_ok())
        mb.add_definition("Init", self._classic_init())

        # Actions
        mb.add_definition("Want", self._classic_want(), params=["p"])
        mb.add_definition("SetTurn", self._classic_set_turn(), params=["p"])
        mb.add_definition("EnterCritical",
                          self._classic_enter_critical(), params=["p"])
        mb.add_definition("ExitCritical",
                          self._classic_exit_critical(), params=["p"])

        mb.add_definition("Next", self._classic_next())
        mb.add_definition("MutualExclusion", self._classic_mutex())
        mb.add_definition("StarvationFreedom",
                          self._classic_starvation_freedom())
        mb.add_definition("EventuallyCritical",
                          self._classic_eventually_critical())

        vars_t = make_vars_tuple(*_ALL_VARS_2)
        fairness = self._classic_fairness(vars_t)
        spec_expr = make_spec_with_fairness("Init", "Next", vars_t, fairness)
        mb.add_definition("Spec", spec_expr)

        for prop in self.get_properties():
            mb.add_property(prop)
        return mb.build()

    # -- Classic helpers -----------------------------------------------

    def _classic_other(self) -> Expression:
        """Other(p) == IF p = 0 THEN 1 ELSE 0"""
        return IfThenElse(
            condition=make_eq(ident("p"), int_lit(0)),
            then_expr=int_lit(1),
            else_expr=int_lit(0),
        )

    def _classic_type_ok(self) -> Expression:
        pc_ok = make_forall_single(
            "p", ident("Proc"),
            make_in(make_func_apply(ident("pc"), ident("p")),
                    make_string_set(*_PC_VALUES)),
        )
        flag_ok = make_forall_single(
            "p", ident("Proc"),
            make_in(make_func_apply(ident("flag"), ident("p")),
                    make_set_enum(bool_lit(True), bool_lit(False))),
        )
        turn_ok = make_in(ident("turn"), ident("Proc"))
        return make_conjunction([pc_ok, flag_ok, turn_ok])

    def _classic_init(self) -> Expression:
        pc_init = make_eq(
            ident("pc"),
            make_function_construction("p", ident("Proc"), str_lit("idle")),
        )
        flag_init = make_eq(
            ident("flag"),
            make_function_construction("p", ident("Proc"), bool_lit(False)),
        )
        turn_init = make_eq(ident("turn"), int_lit(0))
        return make_conjunction([pc_init, flag_init, turn_init])

    def _classic_want(self) -> Expression:
        """Want(p) ==
           /\\ pc[p] = "idle"
           /\\ pc' = [pc EXCEPT ![p] = "want"]
           /\\ flag' = [flag EXCEPT ![p] = TRUE]
           /\\ turn' = Other(p)
           /\\ (effectively sets turn to the other process)
        """
        guard = make_eq(
            make_func_apply(ident("pc"), ident("p")), str_lit("idle"),
        )
        pc_update = make_primed_func_update("pc", ident("p"), str_lit("want"))
        flag_update = make_primed_func_update("flag", ident("p"),
                                              bool_lit(True))
        turn_update = make_primed_eq(
            "turn", _make_user_call("Other", ident("p")),
        )
        return make_conjunction([guard, pc_update, flag_update, turn_update])

    def _classic_set_turn(self) -> Expression:
        """SetTurn(p) ==
           /\\ pc[p] = "want"
           /\\ pc' = [pc EXCEPT ![p] = "wait"]
           /\\ UNCHANGED <<flag, turn>>
        """
        guard = make_eq(
            make_func_apply(ident("pc"), ident("p")), str_lit("want"),
        )
        pc_update = make_primed_func_update("pc", ident("p"), str_lit("wait"))
        return make_guard(guard, [pc_update], ["flag", "turn"])

    def _classic_enter_critical(self) -> Expression:
        """EnterCritical(p) ==
           /\\ pc[p] = "wait"
           /\\ flag[Other(p)] = FALSE \\/ turn = p
           /\\ pc' = [pc EXCEPT ![p] = "critical"]
           /\\ UNCHANGED <<flag, turn>>
        """
        other_p = _make_user_call("Other", ident("p"))
        guard = make_land(
            make_eq(
                make_func_apply(ident("pc"), ident("p")), str_lit("wait"),
            ),
            make_lor(
                make_eq(make_func_apply(ident("flag"), other_p),
                        bool_lit(False)),
                make_eq(ident("turn"), ident("p")),
            ),
        )
        pc_update = make_primed_func_update("pc", ident("p"),
                                            str_lit("critical"))
        return make_guard(guard, [pc_update], ["flag", "turn"])

    def _classic_exit_critical(self) -> Expression:
        """ExitCritical(p) ==
           /\\ pc[p] = "critical"
           /\\ pc' = [pc EXCEPT ![p] = "idle"]
           /\\ flag' = [flag EXCEPT ![p] = FALSE]
           /\\ UNCHANGED turn
        """
        guard = make_eq(
            make_func_apply(ident("pc"), ident("p")), str_lit("critical"),
        )
        pc_update = make_primed_func_update("pc", ident("p"), str_lit("idle"))
        flag_update = make_primed_func_update("flag", ident("p"),
                                              bool_lit(False))
        return make_guard(guard, [pc_update, flag_update], ["turn"])

    def _classic_next(self) -> Expression:
        """Next == \\E p \\in Proc :
             \\/ Want(p) \\/ SetTurn(p) \\/ EnterCritical(p) \\/ ExitCritical(p)
        """
        per_proc = make_disjunction([
            _make_user_call("Want", ident("p")),
            _make_user_call("SetTurn", ident("p")),
            _make_user_call("EnterCritical", ident("p")),
            _make_user_call("ExitCritical", ident("p")),
        ])
        return make_exists_single("p", ident("Proc"), per_proc)

    def _classic_mutex(self) -> Expression:
        """MutualExclusion ==
           \\A p1, p2 \\in Proc :
             (pc[p1] = "critical" /\\ pc[p2] = "critical") => p1 = p2
        """
        p1_crit = make_eq(
            make_func_apply(ident("pc"), ident("p1")), str_lit("critical"),
        )
        p2_crit = make_eq(
            make_func_apply(ident("pc"), ident("p2")), str_lit("critical"),
        )
        return QuantifiedExpr(
            quantifier="forall",
            variables=[("p1", ident("Proc")), ("p2", ident("Proc"))],
            body=make_implies(make_land(p1_crit, p2_crit),
                              make_eq(ident("p1"), ident("p2"))),
        )

    def _classic_starvation_freedom(self) -> Expression:
        """StarvationFreedom ==
           \\A p \\in Proc : pc[p] = "want" ~> pc[p] = "critical"
        """
        return make_forall_single(
            "p", ident("Proc"),
            make_leads_to(
                make_eq(make_func_apply(ident("pc"), ident("p")),
                        str_lit("want")),
                make_eq(make_func_apply(ident("pc"), ident("p")),
                        str_lit("critical")),
            ),
        )

    def _classic_eventually_critical(self) -> Expression:
        """EventuallyCritical ==
           \\A p \\in Proc :
             [](pc[p] = "want" => <>(pc[p] = "critical"))
        """
        return make_forall_single(
            "p", ident("Proc"),
            AlwaysExpr(
                expr=make_implies(
                    make_eq(make_func_apply(ident("pc"), ident("p")),
                            str_lit("want")),
                    EventuallyExpr(
                        expr=make_eq(
                            make_func_apply(ident("pc"), ident("p")),
                            str_lit("critical"),
                        ),
                    ),
                ),
            ),
        )

    def _classic_fairness(self, vars_tuple: Expression) -> List[Expression]:
        flist: List[Expression] = []
        for action in ("Want", "SetTurn", "EnterCritical", "ExitCritical"):
            action_all = make_exists_single(
                "p", ident("Proc"), _make_user_call(action, ident("p")),
            )
            flist.append(make_wf(vars_tuple, action_all))
        return flist

    # ==================================================================
    # Filter Lock (N >= 3)
    # ==================================================================

    def _build_filter_lock_module(self) -> Module:
        mb = ModuleBuilder("PetersonFilterLock")
        mb.add_extends("Naturals")
        mb.add_constants("Proc", "N")
        mb.add_variables(*_ALL_VARS_N)

        # N-1 levels: 1..N-1
        levels_set = make_int_range(int_lit(0), make_minus(ident("N"),
                                                           int_lit(1)))

        mb.add_definition("Levels", levels_set)
        mb.add_definition("PCValues", make_string_set(*_PC_VALUES))
        mb.add_definition("TypeOK", self._filter_type_ok())
        mb.add_definition("Init", self._filter_init())

        # Actions
        mb.add_definition("TryLevel",
                          self._filter_try_level(), params=["p", "lv"])
        mb.add_definition("PassLevel",
                          self._filter_pass_level(), params=["p", "lv"])
        mb.add_definition("EnterCritical",
                          self._filter_enter_critical(), params=["p"])
        mb.add_definition("ExitCritical",
                          self._filter_exit_critical(), params=["p"])
        mb.add_definition("StartWanting",
                          self._filter_start_wanting(), params=["p"])

        mb.add_definition("Next", self._filter_next())
        mb.add_definition("MutualExclusion", self._filter_mutex())
        mb.add_definition("StarvationFreedom",
                          self._filter_starvation_freedom())
        mb.add_definition("EventuallyCritical",
                          self._filter_eventually_critical())

        vars_t = make_vars_tuple(*_ALL_VARS_N)
        fairness = self._filter_fairness(vars_t)
        spec_expr = make_spec_with_fairness("Init", "Next", vars_t, fairness)
        mb.add_definition("Spec", spec_expr)

        for prop in self.get_properties():
            mb.add_property(prop)
        return mb.build()

    # -- Filter lock helpers -------------------------------------------

    def _filter_type_ok(self) -> Expression:
        pc_ok = make_forall_single(
            "p", ident("Proc"),
            make_in(make_func_apply(ident("pc"), ident("p")),
                    make_string_set(*_PC_VALUES)),
        )
        level_ok = make_forall_single(
            "p", ident("Proc"),
            make_in(
                make_func_apply(ident("level"), ident("p")),
                make_int_range(int_lit(0), ident("N")),
            ),
        )
        last_ok = make_forall_single(
            "lv", ident("Levels"),
            make_in(
                make_func_apply(ident("last_to_enter"), ident("lv")),
                ident("Proc"),
            ),
        )
        waiting_ok = make_forall_single(
            "p", ident("Proc"),
            make_in(
                make_func_apply(ident("waiting"), ident("p")),
                make_set_enum(bool_lit(True), bool_lit(False)),
            ),
        )
        return make_conjunction([pc_ok, level_ok, last_ok, waiting_ok])

    def _filter_init(self) -> Expression:
        pc_init = make_eq(
            ident("pc"),
            make_function_construction("p", ident("Proc"), str_lit("idle")),
        )
        level_init = make_eq(
            ident("level"),
            make_function_construction("p", ident("Proc"), int_lit(0)),
        )
        last_init = make_eq(
            ident("last_to_enter"),
            make_function_construction("lv", ident("Levels"), int_lit(0)),
        )
        waiting_init = make_eq(
            ident("waiting"),
            make_function_construction("p", ident("Proc"), bool_lit(False)),
        )
        return make_conjunction([pc_init, level_init, last_init, waiting_init])

    def _filter_start_wanting(self) -> Expression:
        """StartWanting(p) ==
           /\\ pc[p] = "idle"
           /\\ pc' = [pc EXCEPT ![p] = "want"]
           /\\ level' = [level EXCEPT ![p] = 1]
           /\\ waiting' = [waiting EXCEPT ![p] = TRUE]
           /\\ UNCHANGED last_to_enter
        """
        guard = make_eq(
            make_func_apply(ident("pc"), ident("p")), str_lit("idle"),
        )
        pc_upd = make_primed_func_update("pc", ident("p"), str_lit("want"))
        level_upd = make_primed_func_update("level", ident("p"), int_lit(1))
        waiting_upd = make_primed_func_update("waiting", ident("p"),
                                              bool_lit(True))
        return make_guard(guard, [pc_upd, level_upd, waiting_upd],
                          ["last_to_enter"])

    def _filter_try_level(self) -> Expression:
        """TryLevel(p, lv) ==
           /\\ pc[p] = "want"
           /\\ level[p] = lv
           /\\ lv < N - 1
           /\\ last_to_enter' = [last_to_enter EXCEPT ![lv] = p]
           /\\ pc' = [pc EXCEPT ![p] = "wait"]
           /\\ UNCHANGED <<level, waiting>>
        """
        guard = make_conjunction([
            make_eq(make_func_apply(ident("pc"), ident("p")),
                    str_lit("want")),
            make_eq(make_func_apply(ident("level"), ident("p")),
                    ident("lv")),
            make_lt(ident("lv"), make_minus(ident("N"), int_lit(1))),
        ])
        last_upd = make_primed_func_update("last_to_enter", ident("lv"),
                                           ident("p"))
        pc_upd = make_primed_func_update("pc", ident("p"), str_lit("wait"))
        return make_guard(guard, [last_upd, pc_upd], ["level", "waiting"])

    def _filter_pass_level(self) -> Expression:
        """PassLevel(p, lv) ==
           /\\ pc[p] = "wait"
           /\\ level[p] = lv
           /\\ (last_to_enter[lv] # p
                \\/ \\A q \\in Proc \\ {p} : level[q] < lv)
           /\\ level' = [level EXCEPT ![p] = lv + 1]
           /\\ pc' = [pc EXCEPT ![p] = "want"]
           /\\ UNCHANGED <<last_to_enter, waiting>>
        """
        from .spec_utils import make_setdiff
        guard_pc = make_eq(
            make_func_apply(ident("pc"), ident("p")), str_lit("wait"),
        )
        guard_level = make_eq(
            make_func_apply(ident("level"), ident("p")), ident("lv"),
        )
        not_last = make_neq(
            make_func_apply(ident("last_to_enter"), ident("lv")),
            ident("p"),
        )
        all_below = make_forall_single(
            "q",
            make_setdiff(ident("Proc"), make_set_enum(ident("p"))),
            make_lt(
                make_func_apply(ident("level"), ident("q")),
                ident("lv"),
            ),
        )
        pass_condition = make_lor(not_last, all_below)
        guard = make_conjunction([guard_pc, guard_level, pass_condition])
        level_upd = make_primed_func_update(
            "level", ident("p"), make_plus(ident("lv"), int_lit(1)),
        )
        pc_upd = make_primed_func_update("pc", ident("p"), str_lit("want"))
        return make_guard(guard, [level_upd, pc_upd],
                          ["last_to_enter", "waiting"])

    def _filter_enter_critical(self) -> Expression:
        """EnterCritical(p) ==
           /\\ pc[p] = "want"
           /\\ level[p] = N - 1
           /\\ pc' = [pc EXCEPT ![p] = "critical"]
           /\\ waiting' = [waiting EXCEPT ![p] = FALSE]
           /\\ UNCHANGED <<level, last_to_enter>>
        """
        guard = make_conjunction([
            make_eq(make_func_apply(ident("pc"), ident("p")),
                    str_lit("want")),
            make_eq(make_func_apply(ident("level"), ident("p")),
                    make_minus(ident("N"), int_lit(1))),
        ])
        pc_upd = make_primed_func_update("pc", ident("p"),
                                         str_lit("critical"))
        waiting_upd = make_primed_func_update("waiting", ident("p"),
                                              bool_lit(False))
        return make_guard(guard, [pc_upd, waiting_upd],
                          ["level", "last_to_enter"])

    def _filter_exit_critical(self) -> Expression:
        """ExitCritical(p) ==
           /\\ pc[p] = "critical"
           /\\ pc' = [pc EXCEPT ![p] = "idle"]
           /\\ level' = [level EXCEPT ![p] = 0]
           /\\ waiting' = [waiting EXCEPT ![p] = FALSE]
           /\\ UNCHANGED last_to_enter
        """
        guard = make_eq(
            make_func_apply(ident("pc"), ident("p")), str_lit("critical"),
        )
        pc_upd = make_primed_func_update("pc", ident("p"), str_lit("idle"))
        level_upd = make_primed_func_update("level", ident("p"), int_lit(0))
        waiting_upd = make_primed_func_update("waiting", ident("p"),
                                              bool_lit(False))
        return make_guard(guard, [pc_upd, level_upd, waiting_upd],
                          ["last_to_enter"])

    def _filter_next(self) -> Expression:
        """Next == \\E p \\in Proc :
             \\/ StartWanting(p)
             \\/ \\E lv \\in Levels : TryLevel(p, lv) \\/ PassLevel(p, lv)
             \\/ EnterCritical(p)
             \\/ ExitCritical(p)
        """
        level_actions = make_exists_single(
            "lv", ident("Levels"),
            make_disjunction([
                _make_user_call("TryLevel", ident("p"), ident("lv")),
                _make_user_call("PassLevel", ident("p"), ident("lv")),
            ]),
        )
        per_proc = make_disjunction([
            _make_user_call("StartWanting", ident("p")),
            level_actions,
            _make_user_call("EnterCritical", ident("p")),
            _make_user_call("ExitCritical", ident("p")),
        ])
        return make_exists_single("p", ident("Proc"), per_proc)

    def _filter_mutex(self) -> Expression:
        p1_crit = make_eq(
            make_func_apply(ident("pc"), ident("p1")), str_lit("critical"),
        )
        p2_crit = make_eq(
            make_func_apply(ident("pc"), ident("p2")), str_lit("critical"),
        )
        return QuantifiedExpr(
            quantifier="forall",
            variables=[("p1", ident("Proc")), ("p2", ident("Proc"))],
            body=make_implies(make_land(p1_crit, p2_crit),
                              make_eq(ident("p1"), ident("p2"))),
        )

    def _filter_starvation_freedom(self) -> Expression:
        return make_forall_single(
            "p", ident("Proc"),
            make_leads_to(
                make_eq(make_func_apply(ident("pc"), ident("p")),
                        str_lit("want")),
                make_eq(make_func_apply(ident("pc"), ident("p")),
                        str_lit("critical")),
            ),
        )

    def _filter_eventually_critical(self) -> Expression:
        return make_forall_single(
            "p", ident("Proc"),
            AlwaysExpr(
                expr=make_implies(
                    make_eq(make_func_apply(ident("pc"), ident("p")),
                            str_lit("want")),
                    EventuallyExpr(
                        expr=make_eq(
                            make_func_apply(ident("pc"), ident("p")),
                            str_lit("critical"),
                        ),
                    ),
                ),
            ),
        )

    def _filter_fairness(self, vars_tuple: Expression) -> List[Expression]:
        flist: List[Expression] = []
        for action in ("StartWanting", "EnterCritical", "ExitCritical"):
            action_all = make_exists_single(
                "p", ident("Proc"), _make_user_call(action, ident("p")),
            )
            flist.append(make_wf(vars_tuple, action_all))
        # Level actions: WF for combined TryLevel/PassLevel
        level_action = make_exists_single(
            "p", ident("Proc"),
            make_exists_single(
                "lv", ident("Levels"),
                make_disjunction([
                    _make_user_call("TryLevel", ident("p"), ident("lv")),
                    _make_user_call("PassLevel", ident("p"), ident("lv")),
                ]),
            ),
        )
        flist.append(make_wf(vars_tuple, level_action))
        return flist

    # ------------------------------------------------------------------
    # Property wrappers
    # ------------------------------------------------------------------

    def _type_ok_property(self) -> Property:
        return make_invariant_property("TypeOK", ident("TypeOK"))

    def _mutex_property(self) -> Property:
        return make_safety_property("MutualExclusion",
                                    ident("MutualExclusion"))

    def _starvation_freedom_property(self) -> Property:
        return make_liveness_property("StarvationFreedom",
                                     ident("StarvationFreedom"))

    def _eventually_critical_property(self) -> Property:
        return make_temporal_property("EventuallyCritical",
                                     ident("EventuallyCritical"))

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_states(n: int) -> Dict[str, int]:
        pc_states = 4 ** n
        if n == 2:
            flag_states = 2 ** n
            turn_states = n
            total = pc_states * flag_states * turn_states
        else:
            level_states = n ** n  # each process at level 0..N-1
            last_states = n ** (n - 1)
            waiting_states = 2 ** n
            total = pc_states * level_states * last_states * waiting_states
        return {"upper_bound": total,
                "note": "Reachable states are much smaller"}

    @staticmethod
    def supported_configurations() -> List[Dict[str, Any]]:
        return [
            {"name": "small", "n_processes": 2,
             "description": "Classic 2-process Peterson"},
            {"name": "medium", "n_processes": 3,
             "description": "3-process filter lock"},
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
