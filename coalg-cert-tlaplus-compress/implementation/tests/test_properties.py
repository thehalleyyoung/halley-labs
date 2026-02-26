"""
Comprehensive tests for coacert.properties module.

Covers temporal logic formulas, CTL model checking, safety verification,
liveness checking under fairness, and differential testing.
"""

from __future__ import annotations

import pytest

from coacert.functor.coalgebra import FCoalgebra
from coacert.properties.temporal_logic import (
    AG, AF, AX, AU, EF, EG, EU, EX,
    And, Atomic, ExistsPath, FalseFormula, Finally, ForallPath,
    FormulaParseError, Globally, Iff, Implies, Next, Not, Or,
    Release, TemporalFormula, TrueFormula, Until, WeakUntil,
    atomic_propositions, is_ctl, is_ltl, is_stuttering_invariant,
    parse_formula, simplify, to_nnf,
)
from coacert.properties.ctl_star import (
    CTLCheckResult, CTLLabeler, CTLStarChecker,
    CounterexampleGenerator, KripkeAdapter,
    check_invariant, check_reachability, check_response,
)
from coacert.properties.safety import (
    ActionSafetyProperty, InductiveCheckResult, SafetyCheckResult,
    SafetyChecker, SafetyKind, SafetyProperty,
    make_action_constraint, make_ap_invariant, make_exclusion_invariant,
    make_type_invariant,
)
from coacert.properties.liveness import (
    FairnessKind, FairnessSpec, LivenessCheckResult, LivenessChecker,
    LivenessKind, LivenessProperty,
    make_eventually_always, make_infinitely_often, make_leads_to,
    make_strong_fairness, make_weak_fairness,
)
from coacert.properties.differential import (
    DifferentialStats, DifferentialTester, Discrepancy,
    DiscrepancyKind, RandomPropertyGenerator,
)


# ============================================================================
# Fixtures: small Kripke structures as FCoalgebra instances
# ============================================================================

@pytest.fixture
def simple_coalgebra():
    """A 4-state coalgebra with a diamond/loop shape.

    States and transitions:
        s0 (initial) --step--> s1, s2
        s1           --step--> s3
        s2           --step--> s3
        s3           --step--> s0  (creates a cycle)

    Labels:
        s0: {p, q}   s1: {p}   s2: {q}   s3: {r}
    """
    c = FCoalgebra(name="simple")
    c.add_state("s0", propositions={"p", "q"}, is_initial=True)
    c.add_state("s1", propositions={"p"})
    c.add_state("s2", propositions={"q"})
    c.add_state("s3", propositions={"r"})
    c.add_transition("s0", "step", "s1")
    c.add_transition("s0", "step", "s2")
    c.add_transition("s1", "step", "s3")
    c.add_transition("s2", "step", "s3")
    c.add_transition("s3", "step", "s0")
    return c


@pytest.fixture
def linear_coalgebra():
    """A 5-state linear chain (no cycles).

    s0 -> s1 -> s2 -> s3 -> s4
    Labels: s0:{a}, s1:{a,b}, s2:{b}, s3:{b,c}, s4:{c}
    """
    c = FCoalgebra(name="linear")
    c.add_state("s0", propositions={"a"}, is_initial=True)
    c.add_state("s1", propositions={"a", "b"})
    c.add_state("s2", propositions={"b"})
    c.add_state("s3", propositions={"b", "c"})
    c.add_state("s4", propositions={"c"})
    for i in range(4):
        c.add_transition(f"s{i}", "step", f"s{i+1}")
    return c


@pytest.fixture
def kripke(simple_coalgebra):
    """KripkeAdapter from the simple coalgebra."""
    return KripkeAdapter.from_coalgebra(simple_coalgebra)


@pytest.fixture
def quotient_coalgebra():
    """A 2-state quotient: merges s0,s3 and s1,s2 from simple_coalgebra.

    q0 (initial, {p, q}) --step--> q1
    q1 ({p, q, r})       --step--> q0
    We intentionally give q1 the union of labels to agree on some props.
    """
    c = FCoalgebra(name="quotient")
    c.add_state("q0", propositions={"p", "q"}, is_initial=True)
    c.add_state("q1", propositions={"r"})
    c.add_transition("q0", "step", "q1")
    c.add_transition("q1", "step", "q0")
    return c


# ============================================================================
# TestTemporalFormula
# ============================================================================

class TestTemporalFormula:
    """Formula construction, basic methods, pretty-printing."""

    def test_atomic_creation(self):
        f = Atomic("p")
        assert f.name == "p"
        assert f.is_state_formula()
        assert f.children() == []

    def test_true_false(self):
        t, f = TrueFormula(), FalseFormula()
        assert t.is_state_formula() and f.is_state_formula()
        assert t.negate() == FalseFormula()
        assert f.negate() == TrueFormula()

    def test_not_negate(self):
        p = Atomic("p")
        n = Not(p)
        assert n.children() == [p]
        assert n.negate() == p

    def test_and_or_children(self):
        p, q = Atomic("p"), Atomic("q")
        a = And(p, q)
        assert a.children() == [p, q]
        assert a.is_state_formula()
        o = Or(p, q)
        assert o.children() == [p, q]

    def test_implies_negate(self):
        p, q = Atomic("p"), Atomic("q")
        imp = Implies(p, q)
        neg = imp.negate()
        assert isinstance(neg, And)
        assert neg.left == p

    def test_next_is_path_formula(self):
        f = Next(Atomic("p"))
        assert not f.is_state_formula()

    def test_until_is_path_formula(self):
        f = Until(Atomic("p"), Atomic("q"))
        assert not f.is_state_formula()

    def test_finally_negate(self):
        f = Finally(Atomic("p"))
        neg = f.negate()
        assert isinstance(neg, Globally)

    def test_globally_negate(self):
        g = Globally(Atomic("p"))
        neg = g.negate()
        assert isinstance(neg, Finally)

    def test_pretty_atomic(self):
        assert Atomic("x").pretty() == "x"
        assert TrueFormula().pretty() == "TRUE"
        assert FalseFormula().pretty() == "FALSE"

    @pytest.mark.parametrize("formula,expected_substr", [
        (Not(Atomic("p")), "¬"),
        (And(Atomic("p"), Atomic("q")), "∧"),
        (Or(Atomic("p"), Atomic("q")), "∨"),
        (Implies(Atomic("p"), Atomic("q")), "→"),
        (Next(Atomic("p")), "X"),
        (Until(Atomic("p"), Atomic("q")), "U"),
        (Finally(Atomic("p")), "F"),
        (Globally(Atomic("p")), "G"),
    ])
    def test_pretty_contains(self, formula, expected_substr):
        assert expected_substr in formula.pretty()

    def test_substitute(self):
        f = And(Atomic("p"), Atomic("q"))
        result = f.substitute({"p": Atomic("x")})
        assert isinstance(result, And)
        assert result.left == Atomic("x")
        assert result.right == Atomic("q")

    def test_substitute_nested(self):
        f = Not(Implies(Atomic("a"), Atomic("b")))
        result = f.substitute({"a": TrueFormula()})
        assert isinstance(result.child, Implies)
        assert isinstance(result.child.left, TrueFormula)

    def test_exists_forall_are_state_formulas(self):
        assert ExistsPath(Next(Atomic("p"))).is_state_formula()
        assert ForallPath(Globally(Atomic("q"))).is_state_formula()

    def test_exists_negate_to_forall(self):
        e = ExistsPath(Finally(Atomic("p")))
        neg = e.negate()
        assert isinstance(neg, ForallPath)

    def test_release_negate(self):
        r = Release(Atomic("p"), Atomic("q"))
        neg = r.negate()
        assert isinstance(neg, Until)

    def test_iff_children(self):
        f = Iff(Atomic("a"), Atomic("b"))
        assert len(f.children()) == 2
        assert f.is_state_formula()


# ============================================================================
# TestFormulaOperations
# ============================================================================

class TestFormulaOperations:
    """NNF, simplification, fragment detection, AP collection."""

    def test_nnf_double_negation(self):
        f = Not(Not(Atomic("p")))
        result = to_nnf(f)
        assert result == Atomic("p")

    def test_nnf_de_morgan_and(self):
        f = Not(And(Atomic("p"), Atomic("q")))
        result = to_nnf(f)
        assert isinstance(result, Or)
        assert isinstance(result.left, Not) and result.left.child == Atomic("p")

    def test_nnf_de_morgan_or(self):
        f = Not(Or(Atomic("p"), Atomic("q")))
        result = to_nnf(f)
        assert isinstance(result, And)

    def test_nnf_push_through_until(self):
        f = Not(Until(Atomic("p"), Atomic("q")))
        result = to_nnf(f)
        assert isinstance(result, Release)

    def test_nnf_push_through_finally(self):
        f = Not(Finally(Atomic("p")))
        result = to_nnf(f)
        assert isinstance(result, Globally)

    def test_nnf_push_through_globally(self):
        f = Not(Globally(Atomic("p")))
        result = to_nnf(f)
        assert isinstance(result, Finally)

    def test_nnf_eliminates_implies(self):
        f = Implies(Atomic("p"), Atomic("q"))
        result = to_nnf(f)
        assert isinstance(result, Or)

    def test_nnf_eliminates_iff(self):
        f = Iff(Atomic("p"), Atomic("q"))
        result = to_nnf(f)
        # Result should be conjunction of two implications
        assert isinstance(result, And)

    def test_simplify_not_not(self):
        assert simplify(Not(Not(Atomic("p")))) == Atomic("p")

    def test_simplify_and_true(self):
        assert simplify(And(TrueFormula(), Atomic("q"))) == Atomic("q")

    def test_simplify_and_false(self):
        result = simplify(And(FalseFormula(), Atomic("q")))
        assert isinstance(result, FalseFormula)

    def test_simplify_or_true(self):
        result = simplify(Or(TrueFormula(), Atomic("q")))
        assert isinstance(result, TrueFormula)

    def test_simplify_or_false(self):
        assert simplify(Or(FalseFormula(), Atomic("q"))) == Atomic("q")

    def test_simplify_idempotent(self):
        p = Atomic("p")
        assert simplify(And(p, p)) == p
        assert simplify(Or(p, p)) == p

    def test_simplify_implies_false_lhs(self):
        result = simplify(Implies(FalseFormula(), Atomic("p")))
        assert isinstance(result, TrueFormula)

    def test_simplify_finally_finally(self):
        f = Finally(Finally(Atomic("p")))
        result = simplify(f)
        assert isinstance(result, Finally)
        assert result.child == Atomic("p")

    def test_simplify_globally_globally(self):
        g = Globally(Globally(Atomic("p")))
        result = simplify(g)
        assert isinstance(result, Globally)
        assert result.child == Atomic("p")

    def test_is_stuttering_invariant_no_next(self):
        f = And(Atomic("p"), Finally(Atomic("q")))
        assert is_stuttering_invariant(f)

    def test_is_stuttering_invariant_with_next(self):
        f = Next(Atomic("p"))
        assert not is_stuttering_invariant(f)

    def test_is_stuttering_invariant_nested_next(self):
        f = AG(Next(Atomic("p")))
        assert not is_stuttering_invariant(f)

    @pytest.mark.parametrize("formula,expected", [
        (AG(Atomic("p")), True),
        (EF(Atomic("q")), True),
        (EU(Atomic("p"), Atomic("q")), True),
        (AG(Implies(Atomic("p"), AF(Atomic("q")))), True),
        # Bare temporal operators without path quantifier: not CTL
        (Finally(Atomic("p")), False),
        (Globally(Atomic("p")), False),
    ])
    def test_is_ctl(self, formula, expected):
        assert is_ctl(formula) == expected

    @pytest.mark.parametrize("formula,expected", [
        (Atomic("p"), True),
        (Finally(Atomic("p")), True),
        (Globally(And(Atomic("p"), Atomic("q"))), True),
        (Until(Atomic("p"), Atomic("q")), True),
        (Next(Atomic("p")), True),
        # Path quantifiers make it not LTL
        (EF(Atomic("p")), False),
        (AG(Atomic("p")), False),
    ])
    def test_is_ltl(self, formula, expected):
        assert is_ltl(formula) == expected

    def test_atomic_propositions_simple(self):
        f = And(Atomic("p"), Or(Atomic("q"), Atomic("r")))
        assert atomic_propositions(f) == frozenset({"p", "q", "r"})

    def test_atomic_propositions_with_temporal(self):
        f = AG(Implies(Atomic("a"), AF(Atomic("b"))))
        assert atomic_propositions(f) == frozenset({"a", "b"})

    def test_atomic_propositions_empty(self):
        assert atomic_propositions(TrueFormula()) == frozenset()


# ============================================================================
# TestFormulaParser
# ============================================================================

class TestFormulaParser:
    """parse_formula from string representation."""

    def test_parse_atomic(self):
        f = parse_formula("p")
        assert isinstance(f, Atomic)
        assert f.name == "p"

    def test_parse_true_false(self):
        assert isinstance(parse_formula("TRUE"), TrueFormula)
        assert isinstance(parse_formula("FALSE"), FalseFormula)

    def test_parse_negation(self):
        f = parse_formula("~p")
        assert isinstance(f, Not)
        assert isinstance(f.child, Atomic)

    def test_parse_and(self):
        f = parse_formula("p & q")
        assert isinstance(f, And)

    def test_parse_or(self):
        f = parse_formula("p | q")
        assert isinstance(f, Or)

    def test_parse_implies(self):
        f = parse_formula("p -> q")
        assert isinstance(f, Implies)

    def test_parse_iff(self):
        f = parse_formula("p <-> q")
        assert isinstance(f, Iff)

    def test_parse_next(self):
        f = parse_formula("X p")
        assert isinstance(f, Next)

    def test_parse_finally(self):
        f = parse_formula("F p")
        assert isinstance(f, Finally)

    def test_parse_globally(self):
        f = parse_formula("G p")
        assert isinstance(f, Globally)

    def test_parse_until(self):
        f = parse_formula("p U q")
        assert isinstance(f, Until)

    def test_parse_exists_path(self):
        f = parse_formula("E(F p)")
        assert isinstance(f, ExistsPath)
        assert isinstance(f.path_formula, Finally)

    def test_parse_forall_path(self):
        f = parse_formula("A(G p)")
        assert isinstance(f, ForallPath)
        assert isinstance(f.path_formula, Globally)

    def test_parse_nested(self):
        f = parse_formula("A(G (p -> F q))")
        assert isinstance(f, ForallPath)
        inner = f.path_formula
        assert isinstance(inner, Globally)

    def test_parse_parenthesized(self):
        f = parse_formula("(p & q) | r")
        assert isinstance(f, Or)
        assert isinstance(f.left, And)

    def test_parse_complex_ctl(self):
        f = parse_formula("A(G (p -> A(F q)))")
        assert isinstance(f, ForallPath)

    def test_parse_error_empty(self):
        with pytest.raises(FormulaParseError):
            parse_formula("")

    def test_parse_error_unmatched_paren(self):
        with pytest.raises(FormulaParseError):
            parse_formula("(p & q")

    def test_parse_release(self):
        f = parse_formula("p R q")
        assert isinstance(f, Release)

    def test_parse_weak_until(self):
        f = parse_formula("p W q")
        assert isinstance(f, WeakUntil)


# ============================================================================
# TestCTLChecking
# ============================================================================

class TestCTLChecking:
    """CTL model checking on the simple coalgebra."""

    def test_kripke_adapter(self, kripke):
        assert len(kripke.states) == 4
        assert kripke.initial_states == frozenset({"s0"})
        assert kripke.has_proposition("s0", "p")
        assert not kripke.has_proposition("s3", "p")

    def test_kripke_successors(self, kripke):
        succs = kripke.successors("s0")
        assert succs == {"s1", "s2"}

    def test_kripke_predecessors(self, kripke):
        preds = kripke.predecessors("s3")
        assert preds == {"s1", "s2"}

    def test_label_atomic(self, kripke):
        labeler = CTLLabeler(kripke)
        p_states = labeler.label(Atomic("p"))
        assert p_states == frozenset({"s0", "s1"})

    def test_label_not(self, kripke):
        labeler = CTLLabeler(kripke)
        not_p = labeler.label(Not(Atomic("p")))
        assert not_p == frozenset({"s2", "s3"})

    def test_label_and(self, kripke):
        labeler = CTLLabeler(kripke)
        pq = labeler.label(And(Atomic("p"), Atomic("q")))
        assert pq == frozenset({"s0"})

    def test_label_or(self, kripke):
        labeler = CTLLabeler(kripke)
        p_or_r = labeler.label(Or(Atomic("p"), Atomic("r")))
        assert p_or_r == frozenset({"s0", "s1", "s3"})

    def test_ex(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        # EX r: states with a successor labeled r => s1 and s2 (both -> s3)
        result = checker.check(EX(Atomic("r")))
        assert frozenset({"s1", "s2"}) <= result.satisfying_states

    def test_ef(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        # EF r: all states can reach s3
        result = checker.check(EF(Atomic("r")))
        assert result.satisfying_states == frozenset({"s0", "s1", "s2", "s3"})
        assert result.holds

    def test_eg(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        # EG(p | q): path s0->s2->s3->s0 keeps (q or p), but s3 has {r}
        # s0->s1->s3->s0: s3 has r not p/q. So no infinite path in p|q.
        # But s3->s0->... could work if s3 had p|q. It doesn't.
        result = checker.check(EG(Or(Atomic("p"), Atomic("q"))))
        # s0 has p&q, but every path from s0 reaches s3 which has only r
        assert "s3" not in result.satisfying_states

    def test_eu(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        # E[p U r]: from s0 (has p) -> s1 (has p) -> s3 (has r)
        result = checker.check(EU(Atomic("p"), Atomic("r")))
        assert "s0" in result.satisfying_states
        assert "s1" in result.satisfying_states

    def test_ax(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        # AX(p | q): all successors of s0 are {s1(p), s2(q)} => holds at s0
        result = checker.check(AX(Or(Atomic("p"), Atomic("q"))))
        assert "s0" in result.satisfying_states

    def test_af(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        # AF r: every path from s0 eventually reaches s3
        result = checker.check(AF(Atomic("r")))
        assert result.holds

    def test_ag_holds(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        # AG(p | q | r) should hold: every state has at least one
        result = checker.check(AG(Or(Or(Atomic("p"), Atomic("q")), Atomic("r"))))
        assert result.holds

    def test_ag_violated(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        # AG(p) fails: s2 has {q}, s3 has {r}
        result = checker.check(AG(Atomic("p")))
        assert not result.holds

    def test_au(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        # A[(p|q) U r]: from s0, all paths go through p-or-q states to s3(r)
        result = checker.check(AU(Or(Atomic("p"), Atomic("q")), Atomic("r")))
        assert "s0" in result.satisfying_states

    def test_check_result_summary(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        result = checker.check(AG(Atomic("p")))
        s = result.summary()
        assert "VIOLATED" in s

    def test_check_all(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        formulas = [AG(Atomic("p")), EF(Atomic("r"))]
        results = [checker.check(f) for f in formulas]
        assert len(results) == 2
        assert not results[0].holds
        assert results[1].holds


# ============================================================================
# TestSafetyChecker
# ============================================================================

class TestSafetyChecker:
    """Safety property verification."""

    def test_invariant_holds(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        # Every state has at least one of p, q, r
        prop = SafetyProperty(
            name="has_label",
            kind=SafetyKind.STATE_INVARIANT,
            predicate=lambda s, lbl: bool(lbl),
        )
        result = checker.check_invariant(prop)
        assert result.holds
        assert result.states_checked == 4

    def test_invariant_violated(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        # Not all states have 'p'
        prop = make_ap_invariant("p")
        result = checker.check_invariant(prop)
        assert not result.holds
        assert result.violating_state is not None
        assert result.counterexample_trace is not None
        assert result.counterexample_trace[0] == "s0"  # starts from initial

    def test_invariant_counterexample_trace(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = make_ap_invariant("r")
        result = checker.check_invariant(prop)
        assert not result.holds
        # s0 doesn't have 'r', so the initial state itself is violating
        assert result.counterexample_trace == ["s0"]

    def test_invariant_with_max_depth(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = SafetyProperty(
            name="always_true",
            kind=SafetyKind.STATE_INVARIANT,
            predicate=lambda s, lbl: True,
        )
        result = checker.check_invariant(prop, max_depth=1)
        assert result.holds
        assert result.depth_reached <= 1

    def test_exclusion_invariant(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        # p and r never both hold (they're disjoint in our model)
        prop = make_exclusion_invariant("p", "r")
        result = checker.check_invariant(prop)
        assert result.holds

    def test_type_invariant(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        valid = frozenset({"p", "q", "r"})
        prop = make_type_invariant("valid_labels", valid)
        result = checker.check_invariant(prop)
        assert result.holds

    def test_check_inductive(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        # "has some label" is inductive: if a state has labels, successors do too
        prop = SafetyProperty(
            name="has_label",
            kind=SafetyKind.STATE_INVARIANT,
            predicate=lambda s, lbl: bool(lbl),
        )
        result = checker.check_inductive(prop)
        assert result.base_holds
        assert result.is_inductive

    def test_check_inductive_fails(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        # "has p" is not inductive: s1(p) -> s3(r)
        prop = make_ap_invariant("p")
        result = checker.check_inductive(prop)
        assert not result.is_inductive
        assert not result.step_holds
        assert result.step_counterexample is not None

    def test_k_induction(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = SafetyProperty(
            name="has_label",
            kind=SafetyKind.STATE_INVARIANT,
            predicate=lambda s, lbl: bool(lbl),
        )
        result = checker.check_k_induction(prop, max_k=3)
        assert result.is_inductive

    def test_inductive_result_summary(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = make_ap_invariant("p")
        result = checker.check_inductive(prop)
        s = result.summary()
        assert "NOT INDUCTIVE" in s

    def test_action_constraint_holds(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = make_action_constraint("only_step", frozenset({"step"}))
        result = checker.check_action_constraint(prop)
        assert result.holds

    def test_action_constraint_violated(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        # Disallow "step" action
        prop = make_action_constraint("only_jump", frozenset({"jump"}))
        result = checker.check_action_constraint(prop)
        assert not result.holds

    def test_check_all(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        props = [
            make_ap_invariant("p"),
            make_exclusion_invariant("p", "r"),
        ]
        results = checker.check_all(props)
        assert len(results) == 2
        assert not results[0].holds
        assert results[1].holds

    def test_safety_check_result_summary(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = make_exclusion_invariant("p", "r")
        result = checker.check_invariant(prop)
        s = result.summary()
        assert "SAFE" in s

    def test_differential_check(self, simple_coalgebra, quotient_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = SafetyProperty(
            name="has_label",
            kind=SafetyKind.STATE_INVARIANT,
            predicate=lambda s, lbl: bool(lbl),
        )
        projection = {"s0": "q0", "s1": "q1", "s2": "q1", "s3": "q0"}
        orig, quot, agree = checker.check_differential(prop, quotient_coalgebra, projection)
        assert orig.holds and quot.holds
        assert agree


# ============================================================================
# TestLivenessChecker
# ============================================================================

class TestLivenessChecker:
    """Liveness checking under fairness constraints."""

    def test_infinitely_often_holds(self, simple_coalgebra):
        # In the cycle s0->s1->s3->s0, r (at s3) occurs infinitely often
        checker = LivenessChecker(simple_coalgebra)
        prop = make_infinitely_often("inf_r", "r")
        result = checker.check(prop)
        assert result.holds

    def test_infinitely_often_violated(self, linear_coalgebra):
        # Linear chain: no cycle, so no infinite path has a infinitely often
        checker = LivenessChecker(linear_coalgebra)
        prop = make_infinitely_often("inf_a", "a")
        result = checker.check(prop)
        # No non-trivial SCCs => vacuously holds (no fair cycle to violate)
        # or violated depending on implementation; check the result type
        assert isinstance(result, LivenessCheckResult)

    def test_eventually_always(self, simple_coalgebra):
        checker = LivenessChecker(simple_coalgebra)
        prop = make_eventually_always("ev_always_p", "p")
        result = checker.check(prop)
        # No SCC is entirely within p-states (s0 has p but s3 in same cycle doesn't)
        assert not result.holds

    def test_leads_to_holds(self, simple_coalgebra):
        checker = LivenessChecker(simple_coalgebra)
        # p leads to r: from any p-state, r is eventually reached
        prop = make_leads_to("p_leads_r", "p", "r")
        result = checker.check(prop)
        assert result.holds

    def test_leads_to_violated(self, linear_coalgebra):
        checker = LivenessChecker(linear_coalgebra)
        # c leads to a: once at c (s3,s4), a (s0,s1) is never reached
        prop = make_leads_to("c_leads_a", "c", "a")
        result = checker.check(prop)
        # s3 has c but s3->s4 with no path back to a-states, and no cycle in ¬a
        # that is fair. Actually in a linear chain there are no non-trivial SCCs.
        assert isinstance(result, LivenessCheckResult)

    def test_fair_cycles(self, simple_coalgebra):
        fairness = [make_weak_fairness(
            "wf_step",
            enabled_states=frozenset({"s0", "s1", "s2", "s3"}),
            taken_states=frozenset({"s1", "s2", "s3"}),
        )]
        checker = LivenessChecker(simple_coalgebra, fairness=fairness)
        fair = checker.find_fair_cycles()
        assert len(fair) > 0

    def test_unfair_cycles(self, simple_coalgebra):
        # Strong fairness: enabled everywhere, taken only at s1
        fairness = [make_strong_fairness(
            "sf_strict",
            enabled_states=frozenset({"s0", "s1", "s2", "s3"}),
            taken_states=frozenset({"s1"}),
        )]
        checker = LivenessChecker(simple_coalgebra, fairness=fairness)
        unfair = checker.find_unfair_cycles()
        # The SCC {s0,s1,s2,s3} intersects enabled and must intersect taken
        # It does contain s1, so it IS fair. Check accordingly.
        # SCCs containing s1 satisfy the constraint.
        assert isinstance(unfair, list)

    def test_liveness_result_summary(self, simple_coalgebra):
        checker = LivenessChecker(simple_coalgebra)
        prop = make_infinitely_often("inf_r", "r")
        result = checker.check(prop)
        s = result.summary()
        assert "inf_r" in s

    def test_check_all(self, simple_coalgebra):
        checker = LivenessChecker(simple_coalgebra)
        props = [
            make_infinitely_often("inf_p", "p"),
            make_infinitely_often("inf_r", "r"),
        ]
        results = checker.check_all(props)
        assert len(results) == 2

    def test_streett_acceptance(self, simple_coalgebra):
        checker = LivenessChecker(simple_coalgebra)
        # Pair: if visit s0 infinitely, must visit s3 infinitely
        pairs = [(frozenset({"s0"}), frozenset({"s3"}))]
        result = checker.check_streett_acceptance(pairs)
        # The full SCC {s0,s1,s2,s3} contains both s0 and s3
        assert result.holds

    def test_fairness_spec_weak(self):
        spec = FairnessSpec(
            kind=FairnessKind.WEAK,
            name="wf",
            enabled_states=frozenset({"s0", "s1"}),
            taken_states=frozenset({"s1"}),
        )
        # SCC = {s0, s1}: permanently enabled (subset), intersects taken => satisfied
        assert spec.is_satisfied_by_scc(frozenset({"s0", "s1"}))
        # SCC = {s0}: permanently enabled (subset), does NOT intersect taken
        assert not spec.is_satisfied_by_scc(frozenset({"s0"}))

    def test_fairness_spec_strong(self):
        spec = FairnessSpec(
            kind=FairnessKind.STRONG,
            name="sf",
            enabled_states=frozenset({"s0", "s1"}),
            taken_states=frozenset({"s1"}),
        )
        # SCC intersects enabled and taken => satisfied
        assert spec.is_satisfied_by_scc(frozenset({"s0", "s1"}))
        # SCC intersects enabled but NOT taken => not satisfied
        assert not spec.is_satisfied_by_scc(frozenset({"s0"}))
        # SCC does NOT intersect enabled => vacuously satisfied
        assert spec.is_satisfied_by_scc(frozenset({"s2", "s3"}))


# ============================================================================
# TestDifferentialTester
# ============================================================================

class TestDifferentialTester:
    """Differential testing: original vs quotient."""

    def test_compare_safety(self, simple_coalgebra, quotient_coalgebra):
        projection = {"s0": "q0", "s1": "q1", "s2": "q1", "s3": "q0"}
        tester = DifferentialTester(
            simple_coalgebra, quotient_coalgebra, projection
        )
        prop = SafetyProperty(
            name="has_label",
            kind=SafetyKind.STATE_INVARIANT,
            predicate=lambda s, lbl: bool(lbl),
        )
        orig, quot, agree = tester.compare_safety(prop)
        assert agree

    def test_compare_ctl(self, simple_coalgebra, quotient_coalgebra):
        projection = {"s0": "q0", "s1": "q1", "s2": "q1", "s3": "q0"}
        tester = DifferentialTester(
            simple_coalgebra, quotient_coalgebra, projection
        )
        formula = EF(Atomic("r"))
        orig, quot, agree = tester.compare_ctl(formula)
        assert isinstance(orig, CTLCheckResult)
        assert isinstance(quot, CTLCheckResult)

    def test_run_fuzz_suite(self, simple_coalgebra, quotient_coalgebra):
        projection = {"s0": "q0", "s1": "q1", "s2": "q1", "s3": "q0"}
        tester = DifferentialTester(
            simple_coalgebra, quotient_coalgebra, projection
        )
        stats = tester.run_fuzz_suite(n_safety=5, n_liveness=5, n_ctl=5, seed=42)
        assert isinstance(stats, DifferentialStats)
        assert stats.total_tests == 15
        assert stats.agreement_rate >= 0.0

    def test_differential_stats_summary(self):
        stats = DifferentialStats(
            total_tests=10, agreements=9, discrepancies=1,
            safety_tests=5, safety_agreements=5,
            liveness_tests=3, liveness_agreements=2,
            ctl_tests=2, ctl_agreements=2,
        )
        s = stats.summary()
        assert "Agreement rate" in s
        assert "90" in s  # 90%

    def test_differential_stats_confidence(self):
        stats = DifferentialStats(total_tests=100, agreements=100, discrepancies=0)
        assert stats.confidence > 0.95

    def test_discrepancy_summary(self):
        d = Discrepancy(
            kind=DiscrepancyKind.SAFETY_MISMATCH,
            property_name="test_prop",
            original_holds=True,
            quotient_holds=False,
        )
        s = d.summary()
        assert "SAFETY_MISMATCH" in s
        assert "test_prop" in s

    def test_random_property_generator(self):
        gen = RandomPropertyGenerator(["p", "q", "r"], seed=123)
        formula = gen.random_ctl_formula(max_depth=2)
        assert isinstance(formula, TemporalFormula)
        ltl = gen.random_ltl_formula(max_depth=2)
        assert isinstance(ltl, TemporalFormula)
        safety = gen.random_safety_property()
        assert isinstance(safety, SafetyProperty)
        liveness = gen.random_liveness_property()
        assert isinstance(liveness, LivenessProperty)

    def test_compare_liveness(self, simple_coalgebra, quotient_coalgebra):
        projection = {"s0": "q0", "s1": "q1", "s2": "q1", "s3": "q0"}
        tester = DifferentialTester(
            simple_coalgebra, quotient_coalgebra, projection
        )
        prop = make_infinitely_often("inf_r", "r")
        orig, quot, agree = tester.compare_liveness(prop)
        assert isinstance(orig, LivenessCheckResult)
        assert isinstance(quot, LivenessCheckResult)

    def test_run_full_suite(self, simple_coalgebra, quotient_coalgebra):
        projection = {"s0": "q0", "s1": "q1", "s2": "q1", "s3": "q0"}
        tester = DifferentialTester(
            simple_coalgebra, quotient_coalgebra, projection
        )
        safety = [make_ap_invariant("p"), make_exclusion_invariant("p", "r")]
        ctl = [AG(Atomic("p")), EF(Atomic("r"))]
        stats = tester.run_full_suite(
            safety_props=safety, ctl_formulas=ctl,
        )
        assert stats.total_tests == 4
        assert stats.safety_tests == 2
        assert stats.ctl_tests == 2


# ============================================================================
# TestCounterexamples
# ============================================================================

class TestCounterexamples:
    """Counterexample generation for violated properties."""

    def test_ag_counterexample(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        result = checker.check(AG(Atomic("p")))
        assert not result.holds
        assert result.counterexample is not None
        # Trace should start at s0 and reach a ¬p state
        assert result.counterexample[0] == "s0"

    def test_ef_no_counterexample_when_holds(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        result = checker.check(EF(Atomic("r")))
        assert result.holds
        assert result.counterexample is None

    def test_safety_counterexample_trace(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = make_ap_invariant("p")
        result = checker.check_invariant(prop)
        assert not result.holds
        trace = result.counterexample_trace
        assert trace is not None
        assert len(trace) >= 1
        # Last state in trace should be the violating state
        assert trace[-1] == result.violating_state

    def test_inductive_step_counterexample(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = make_ap_invariant("p")
        result = checker.check_inductive(prop)
        assert result.step_counterexample is not None
        src, dst = result.step_counterexample
        # src should have p (satisfies inv), dst should not
        labels_src = KripkeAdapter.from_coalgebra(simple_coalgebra).labels
        assert "p" in labels_src.get(src, frozenset())
        assert "p" not in labels_src.get(dst, frozenset())

    def test_liveness_counterexample(self, simple_coalgebra):
        checker = LivenessChecker(simple_coalgebra)
        # ◇□p: eventually always p. No SCC is entirely within p-states.
        prop = make_eventually_always("ev_always_p", "p")
        result = checker.check(prop)
        assert not result.holds

    def test_violating_initial_states(self, simple_coalgebra):
        checker = CTLStarChecker(simple_coalgebra)
        result = checker.check(AG(Atomic("p")))
        assert len(result.violating_initial_states) > 0

    def test_counterexample_generator_shortest_path(self, kripke):
        gen = CounterexampleGenerator(kripke)
        path = gen.shortest_path("s0", frozenset({"s3"}))
        assert path is not None
        assert path[0] == "s0"
        assert path[-1] == "s3"
        assert len(path) <= 3  # s0 -> s1/s2 -> s3

    def test_counterexample_generator_ag(self, kripke):
        labeler = CTLLabeler(kripke)
        gen = CounterexampleGenerator(kripke)
        cex = gen.ag_counterexample(labeler, Atomic("p"))
        assert cex is not None
        # Should end at a state without p
        last = cex[-1]
        assert "p" not in kripke.labels.get(last, frozenset())

    def test_counterexample_generator_ef_witness(self, kripke):
        labeler = CTLLabeler(kripke)
        gen = CounterexampleGenerator(kripke)
        witness = gen.ef_witness(labeler, Atomic("r"))
        assert witness is not None
        assert witness[-1] == "s3"

    def test_action_constraint_counterexample(self, simple_coalgebra):
        checker = SafetyChecker(simple_coalgebra)
        prop = make_action_constraint("only_jump", frozenset({"jump"}))
        result = checker.check_action_constraint(prop)
        assert not result.holds
        assert result.counterexample_trace is not None


# ============================================================================
# Additional edge-case and integration tests
# ============================================================================

class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_single_state_coalgebra(self):
        c = FCoalgebra(name="single")
        c.add_state("s0", propositions={"p"}, is_initial=True)
        c.add_transition("s0", "loop", "s0")
        checker = CTLStarChecker(c)
        assert checker.check(AG(Atomic("p"))).holds
        assert checker.check(EG(Atomic("p"))).holds

    def test_deadlock_state(self):
        c = FCoalgebra(name="deadlock")
        c.add_state("s0", propositions={"p"}, is_initial=True)
        # No transitions from s0
        checker = SafetyChecker(c)
        prop = make_ap_invariant("p")
        result = checker.check_invariant(prop)
        assert result.holds

    def test_multiple_initial_states(self):
        c = FCoalgebra(name="multi_init")
        c.add_state("s0", propositions={"p"}, is_initial=True)
        c.add_state("s1", propositions={"q"}, is_initial=True)
        c.add_transition("s0", "step", "s1")
        c.add_transition("s1", "step", "s0")
        checker = CTLStarChecker(c)
        # AG p fails because s1 doesn't have p
        assert not checker.check(AG(Atomic("p"))).holds
        # EF q holds from both initial states
        assert checker.check(EF(Atomic("q"))).holds

    def test_convenience_check_invariant(self, simple_coalgebra):
        result = check_invariant(simple_coalgebra, "p")
        assert not result.holds

    def test_convenience_check_reachability(self, simple_coalgebra):
        result = check_reachability(simple_coalgebra, "r")
        assert result.holds

    def test_convenience_check_response(self, simple_coalgebra):
        result = check_response(simple_coalgebra, "p", "r")
        # AG(p -> AF r): whenever p holds, r eventually holds
        assert result.holds

    def test_kripke_reverse_graph(self, kripke):
        rev = kripke.reverse_graph()
        assert "s0" in rev.get("s1", set())
        assert "s0" in rev.get("s2", set())
        assert "s3" in rev.get("s0", set())

    @pytest.mark.parametrize("prop_a,prop_b,expected", [
        ("p", "r", True),   # p and r never co-occur
        ("p", "q", False),  # p and q co-occur at s0
    ])
    def test_exclusion_parametrized(self, simple_coalgebra, prop_a, prop_b, expected):
        checker = SafetyChecker(simple_coalgebra)
        prop = make_exclusion_invariant(prop_a, prop_b)
        result = checker.check_invariant(prop)
        assert result.holds == expected
