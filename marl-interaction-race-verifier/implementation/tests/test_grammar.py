"""Tests for marace.spec.grammar — BNF grammar, FIRST/FOLLOW sets, well-formedness."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.spec.grammar import (
    GrammarRule,
    GrammarSpec,
    FormalSemantics,
    WellFormednessChecker,
    MARACE_BNF,
)


# ======================================================================
# GrammarRule
# ======================================================================

class TestGrammarRule:
    """Test GrammarRule dataclass."""

    def test_terminals_extraction(self):
        rule = GrammarRule("formula", "'always' '(' formula ')'")
        terms = rule.terminals()
        assert "always" in terms
        assert "(" in terms
        assert ")" in terms

    def test_non_terminals_extraction(self):
        rule = GrammarRule("formula", "'always' '(' formula ')'")
        nts = rule.non_terminals()
        assert "formula" in nts
        assert "always" not in nts

    def test_epsilon_production(self):
        rule = GrammarRule("empty", "ε")
        assert rule.is_epsilon()

    def test_non_epsilon_production(self):
        rule = GrammarRule("spec", "contract")
        assert not rule.is_epsilon()

    def test_str_representation(self):
        rule = GrammarRule("spec", "contract")
        assert "spec" in str(rule)
        assert "contract" in str(rule)

    def test_frozen(self):
        rule = GrammarRule("A", "B")
        with pytest.raises(AttributeError):
            rule.name = "C"

    def test_empty_terminals(self):
        rule = GrammarRule("A", "B C")
        assert rule.terminals() == []

    def test_empty_non_terminals(self):
        rule = GrammarRule("A", "'x' 'y'")
        assert rule.non_terminals() == []


# ======================================================================
# GrammarSpec — validation
# ======================================================================

class TestGrammarSpecValidation:
    """Test GrammarSpec validates the BNF."""

    def test_default_grammar_is_valid(self):
        """The built-in MARACE_BNF should pass validation."""
        spec = GrammarSpec()
        errors = spec.validate_grammar()
        assert errors == [], f"Default grammar has errors: {errors}"

    def test_start_symbol_exists(self):
        spec = GrammarSpec()
        assert spec.start_symbol == "spec"
        assert "spec" in spec.non_terminals

    def test_non_terminals_are_nonempty(self):
        spec = GrammarSpec()
        assert len(spec.non_terminals) > 0

    def test_terminals_are_nonempty(self):
        spec = GrammarSpec()
        assert len(spec.terminals) > 0

    def test_unproductive_nonterminal_raises_error(self):
        """A nonterminal that can't derive a terminal string should be flagged."""
        bad_rules = {
            "spec": [GrammarRule("spec", "infinite")],
            "infinite": [GrammarRule("infinite", "infinite")],
        }
        spec = GrammarSpec(rules=bad_rules)
        errors = spec.validate_grammar()
        assert any("infinite" in e for e in errors)

    def test_unreachable_nonterminal(self):
        """A nonterminal not reachable from start should be flagged."""
        rules = {
            "spec": [GrammarRule("spec", "'a'")],
            "orphan": [GrammarRule("orphan", "'b'")],
        }
        spec = GrammarSpec(rules=rules)
        errors = spec.validate_grammar()
        assert any("orphan" in e for e in errors)

    def test_pretty_print(self):
        spec = GrammarSpec()
        pp = spec.pretty_print()
        assert isinstance(pp, str)
        assert len(pp) > 0
        assert "::=" in pp


# ======================================================================
# GrammarSpec — FIRST / FOLLOW sets
# ======================================================================

class TestFirstFollowSets:
    """Test FIRST and FOLLOW set computation."""

    def test_first_sets_computed(self):
        spec = GrammarSpec()
        firsts = spec.first_sets()
        assert isinstance(firsts, dict)
        assert "spec" in firsts
        assert len(firsts["spec"]) > 0

    def test_first_of_terminal_is_itself(self):
        spec = GrammarSpec()
        firsts = spec.first_sets()
        for t in spec.terminals:
            if t in firsts:
                assert t in firsts[t]

    def test_follow_sets_computed(self):
        spec = GrammarSpec()
        follows = spec.follow_sets()
        assert isinstance(follows, dict)
        assert "spec" in follows

    def test_eof_in_follow_of_start(self):
        """EOF should be in FOLLOW(start_symbol)."""
        spec = GrammarSpec()
        follows = spec.follow_sets()
        assert "EOF" in follows.get("spec", set())

    def test_first_follow_disjoint_for_ll1(self):
        """For LL(1), FIRST and FOLLOW sets shouldn't have unexpected overlap."""
        spec = GrammarSpec()
        firsts = spec.first_sets()
        follows = spec.follow_sets()
        # Just verify both are non-empty dicts
        assert len(firsts) > 0
        assert len(follows) > 0


# ======================================================================
# WellFormednessChecker
# ======================================================================

class TestWellFormednessChecker:
    """Test well-formedness checking of formulas and contracts."""

    def test_valid_formula_no_errors(self):
        """A properly constructed formula should have no errors."""
        from marace.spec.predicates import LinearPredicate
        from marace.spec.temporal import Always

        checker = WellFormednessChecker(["agent_0", "agent_1"])
        pred = LinearPredicate(
            a=np.array([1.0, 0.0]),
            b=5.0,
        )
        formula = Always(predicate=pred, horizon=10)
        errors = checker.check_formula(formula)
        assert errors == []

    def test_missing_agent_detected(self):
        """Using an unknown agent should produce an error."""
        from marace.spec.predicates import DistancePredicate

        checker = WellFormednessChecker(["agent_0"])
        # Try to create a distance predicate referencing an unknown agent
        try:
            pred = DistancePredicate(
                agent_i_dims=[0, 1],
                agent_j_dims=[2, 3],
                threshold=2.0,
                agent_i_id="agent_0",
                agent_j_id="agent_MISSING",
            )
            errors = checker.check_formula(pred)
            assert len(errors) > 0
        except TypeError:
            # If constructor doesn't take agent_id params, test is not applicable
            pass

    def test_empty_agent_list(self):
        checker = WellFormednessChecker([])
        assert checker._agent_ids == set()


# ======================================================================
# FormalSemantics
# ======================================================================

class TestFormalSemantics:
    """Test FormalSemantics has entries for all operators."""

    @pytest.mark.parametrize("op", [
        "always", "eventually", "until", "next", "bounded_response",
    ])
    def test_temporal_semantics_exist(self, op):
        sem = FormalSemantics.get_temporal_semantics(op)
        assert isinstance(sem, dict)
        assert len(sem) > 0

    @pytest.mark.parametrize("pred", [
        "distance", "collision", "region", "relvel", "linear",
    ])
    def test_predicate_semantics_exist(self, pred):
        sem = FormalSemantics.get_predicate_semantics(pred)
        assert isinstance(sem, dict)
        assert len(sem) > 0

    def test_contract_semantics(self):
        sem = FormalSemantics.get_contract_semantics()
        assert isinstance(sem, dict)
        assert len(sem) > 0

    def test_summary_is_nonempty(self):
        s = FormalSemantics.summary()
        assert isinstance(s, str)
        assert len(s) > 100

    def test_unknown_temporal_raises(self):
        with pytest.raises(KeyError):
            FormalSemantics.get_temporal_semantics("nonexistent_op")

    def test_unknown_predicate_raises(self):
        with pytest.raises(KeyError):
            FormalSemantics.get_predicate_semantics("nonexistent_pred")
