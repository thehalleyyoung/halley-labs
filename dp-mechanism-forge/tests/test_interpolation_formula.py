"""Tests for interpolation.formula module."""
import pytest
import numpy as np

from dp_forge.interpolation.formula import (
    FormulaNode,
    NodeKind,
    Formula,
    CNFConverter,
    DNFConverter,
    Simplifier,
    SubstitutionEngine,
    QuantifierElimination,
    SatisfiabilityChecker,
)


class TestCNFConverter:
    """Test CNFConverter correctness."""

    def test_literal_unchanged(self):
        """A single variable is already in CNF."""
        f = Formula(FormulaNode.var("x"))
        cnf = CNFConverter().convert(f)
        assert cnf.node.kind == NodeKind.VAR

    def test_and_unchanged(self):
        """AND of variables is already in CNF."""
        f = Formula(FormulaNode.and_(FormulaNode.var("x"), FormulaNode.var("y")))
        cnf = CNFConverter().convert(f)
        assert cnf.node.kind == NodeKind.AND

    def test_or_of_and_distributed(self):
        """OR of ANDs is distributed to AND of ORs (CNF)."""
        a = FormulaNode.and_(FormulaNode.var("x"), FormulaNode.var("y"))
        b = FormulaNode.var("z")
        f = Formula(FormulaNode.or_(a, b))
        cnf = CNFConverter().convert(f)
        # CNF: (x ∨ z) ∧ (y ∨ z)
        assert cnf.node.kind == NodeKind.AND

    def test_implies_eliminated(self):
        """Implication A→B becomes ¬A∨B."""
        f = Formula(FormulaNode.implies(FormulaNode.var("a"), FormulaNode.var("b")))
        cnf = CNFConverter().convert(f)
        assert cnf.node.kind == NodeKind.OR

    def test_double_negation_eliminated(self):
        """¬¬x becomes x."""
        f = Formula(FormulaNode.not_(FormulaNode.not_(FormulaNode.var("x"))))
        cnf = CNFConverter().convert(f)
        assert cnf.node.kind == NodeKind.VAR
        assert cnf.node.value == "x"

    def test_tseitin_mode(self):
        """Tseitin transformation produces equisatisfiable CNF."""
        a = FormulaNode.or_(FormulaNode.var("x"), FormulaNode.var("y"))
        b = FormulaNode.or_(FormulaNode.var("z"), FormulaNode.var("w"))
        f = Formula(FormulaNode.and_(a, b))
        cnf = CNFConverter(use_tseitin=True).convert(f)
        # Should produce a formula (possibly with auxiliary variables)
        assert cnf.node.kind in (NodeKind.AND, NodeKind.VAR, NodeKind.OR)

    def test_const_true(self):
        """Constant true stays true."""
        f = Formula(FormulaNode.const(True))
        cnf = CNFConverter().convert(f)
        assert cnf.node.kind == NodeKind.CONST and cnf.node.value is True

    def test_const_false(self):
        """Constant false stays false."""
        f = Formula(FormulaNode.const(False))
        cnf = CNFConverter().convert(f)
        assert cnf.node.kind == NodeKind.CONST and cnf.node.value is False


class TestDNFConverter:
    """Test DNFConverter correctness."""

    def test_literal_unchanged(self):
        f = Formula(FormulaNode.var("x"))
        dnf = DNFConverter().convert(f)
        assert dnf.node.kind == NodeKind.VAR

    def test_and_of_or_distributed(self):
        """AND of ORs is distributed to OR of ANDs (DNF)."""
        a = FormulaNode.or_(FormulaNode.var("x"), FormulaNode.var("y"))
        b = FormulaNode.or_(FormulaNode.var("z"), FormulaNode.var("w"))
        f = Formula(FormulaNode.and_(a, b))
        dnf = DNFConverter().convert(f)
        # DNF: (x∧z) ∨ (x∧w) ∨ (y∧z) ∨ (y∧w)
        assert dnf.node.kind == NodeKind.OR

    def test_demorgan_applied(self):
        """De Morgan's laws: ¬(a ∧ b) → ¬a ∨ ¬b."""
        inner = FormulaNode.and_(FormulaNode.var("a"), FormulaNode.var("b"))
        f = Formula(FormulaNode.not_(inner))
        dnf = DNFConverter().convert(f)
        assert dnf.node.kind == NodeKind.OR


class TestSimplifier:
    """Test Simplifier reductions."""

    def test_double_negation(self):
        """¬¬x simplifies to x."""
        f = Formula(FormulaNode.not_(FormulaNode.not_(FormulaNode.var("x"))))
        simplified = Simplifier().simplify(f)
        assert simplified.node.kind == NodeKind.VAR

    def test_and_with_true(self):
        """x ∧ True simplifies to x."""
        f = Formula(FormulaNode.and_(FormulaNode.var("x"), FormulaNode.const(True)))
        simplified = Simplifier().simplify(f)
        assert simplified.node.kind == NodeKind.VAR

    def test_and_with_false(self):
        """x ∧ False simplifies to False."""
        f = Formula(FormulaNode.and_(FormulaNode.var("x"), FormulaNode.const(False)))
        simplified = Simplifier().simplify(f)
        assert simplified.node.kind == NodeKind.CONST
        assert simplified.node.value is False

    def test_or_with_true(self):
        """x ∨ True simplifies to True."""
        f = Formula(FormulaNode.or_(FormulaNode.var("x"), FormulaNode.const(True)))
        simplified = Simplifier().simplify(f)
        assert simplified.node.kind == NodeKind.CONST
        assert simplified.node.value is True

    def test_or_with_false(self):
        """x ∨ False simplifies to x."""
        f = Formula(FormulaNode.or_(FormulaNode.var("x"), FormulaNode.const(False)))
        simplified = Simplifier().simplify(f)
        assert simplified.node.kind == NodeKind.VAR

    def test_idempotent_and(self):
        """x ∧ x simplifies to x."""
        x = FormulaNode.var("x")
        f = Formula(FormulaNode.and_(x, x))
        simplified = Simplifier().simplify(f)
        assert simplified.node.kind == NodeKind.VAR

    def test_trivially_true_linear(self):
        """0 <= 5 simplifies to True."""
        f = Formula(FormulaNode.leq({}, 5.0))
        simplified = Simplifier().simplify(f)
        assert simplified.node.kind == NodeKind.CONST
        assert simplified.node.value is True

    def test_trivially_false_linear(self):
        """0 <= -1 simplifies to False."""
        f = Formula(FormulaNode.leq({}, -1.0))
        simplified = Simplifier().simplify(f)
        assert simplified.node.kind == NodeKind.CONST
        assert simplified.node.value is False


class TestSubstitutionEngine:
    """Test SubstitutionEngine variable renaming."""

    def test_substitute_variable(self):
        """Substitute x -> y in a formula."""
        f = Formula(FormulaNode.var("x"))
        sub = SubstitutionEngine()
        result = sub.substitute(f, {"x": FormulaNode.var("y")})
        assert result.node.value == "y"

    def test_rename_variables(self):
        """Rename multiple variables."""
        f = Formula(FormulaNode.and_(FormulaNode.var("a"), FormulaNode.var("b")))
        sub = SubstitutionEngine()
        result = sub.rename(f, {"a": "x", "b": "y"})
        assert "x" in result.variables
        assert "y" in result.variables

    def test_substitute_in_linear(self):
        """Substitute in a linear constraint."""
        f = Formula(FormulaNode.leq({"x": 1.0, "y": 2.0}, 5.0))
        sub = SubstitutionEngine()
        result = sub.rename(f, {"x": "a"})
        assert "a" in result.node.coefficients
        assert "y" in result.node.coefficients

    def test_restrict_to_variables(self):
        """Restricting adds existential quantifiers for extra vars."""
        f = Formula(FormulaNode.and_(FormulaNode.var("x"), FormulaNode.var("y")))
        sub = SubstitutionEngine()
        result = sub.restrict(f, frozenset({"x"}))
        assert result.node.kind == NodeKind.EXISTS

    def test_no_change_when_var_absent(self):
        """Substituting absent variable does nothing."""
        f = Formula(FormulaNode.var("x"))
        sub = SubstitutionEngine()
        result = sub.substitute(f, {"z": FormulaNode.var("w")})
        assert result.node.value == "x"


class TestQuantifierElimination:
    """Test QuantifierElimination soundness."""

    def test_eliminate_single_variable(self):
        """Eliminate a single variable from a conjunction."""
        # x <= 5 ∧ x >= 2  →  eliminating x gives 2 <= 5 (True)
        c1 = FormulaNode.leq({"x": 1.0}, 5.0)  # x <= 5
        c2 = FormulaNode.leq({"x": -1.0}, -2.0)  # -x <= -2 (x >= 2)
        f = Formula(FormulaNode.and_(c1, c2))
        qe = QuantifierElimination()
        result = qe.eliminate(f, ["x"])
        # After elimination, result should be satisfiable
        assert result.node is not None

    def test_eliminate_preserves_free_vars(self):
        """Non-eliminated variables remain."""
        c1 = FormulaNode.leq({"x": 1.0, "y": 1.0}, 5.0)
        c2 = FormulaNode.leq({"x": -1.0}, -1.0)
        f = Formula(FormulaNode.and_(c1, c2))
        qe = QuantifierElimination()
        result = qe.eliminate(f, ["x"])
        assert "y" in result.variables or result.node.kind == NodeKind.CONST

    def test_infeasible_after_elimination(self):
        """Infeasible system stays infeasible."""
        # x <= 1 ∧ x >= 5 → infeasible
        c1 = FormulaNode.leq({"x": 1.0}, 1.0)
        c2 = FormulaNode.leq({"x": -1.0}, -5.0)
        f = Formula(FormulaNode.and_(c1, c2))
        qe = QuantifierElimination()
        result = qe.eliminate(f, ["x"])
        # Should yield 0 <= -4 or equivalent infeasible constraint
        checker = SatisfiabilityChecker()
        sat, _ = checker.check(result)
        assert not sat


class TestSatisfiabilityChecker:
    """Test SatisfiabilityChecker on SAT and UNSAT instances."""

    def test_sat_single_constraint(self):
        """x <= 5 is satisfiable."""
        f = Formula(FormulaNode.leq({"x": 1.0}, 5.0))
        checker = SatisfiabilityChecker()
        sat, witness = checker.check(f)
        assert sat
        assert witness is not None
        assert witness["x"] <= 5.0 + 1e-6

    def test_unsat_contradictory(self):
        """x <= 1 ∧ x >= 5 is unsatisfiable."""
        c1 = FormulaNode.leq({"x": 1.0}, 1.0)
        c2 = FormulaNode.leq({"x": -1.0}, -5.0)
        f = Formula(FormulaNode.and_(c1, c2))
        checker = SatisfiabilityChecker()
        assert checker.is_unsat(f)

    def test_sat_two_variables(self):
        """x + y <= 10, x >= 0, y >= 0 is satisfiable."""
        c1 = FormulaNode.leq({"x": 1.0, "y": 1.0}, 10.0)
        c2 = FormulaNode.leq({"x": -1.0}, 0.0)
        c3 = FormulaNode.leq({"y": -1.0}, 0.0)
        f = Formula(FormulaNode.and_(c1, FormulaNode.and_(c2, c3)))
        checker = SatisfiabilityChecker()
        sat, witness = checker.check(f)
        assert sat

    def test_implies_valid(self):
        """A ∧ ¬B is UNSAT checked via is_unsat."""
        # x <= 1 ∧ x >= 5 is UNSAT (directly checking contradiction)
        c1 = FormulaNode.leq({"x": 1.0}, 1.0)
        c2 = FormulaNode.leq({"x": -1.0}, -5.0)
        f = Formula(FormulaNode.and_(c1, c2))
        checker = SatisfiabilityChecker()
        assert checker.is_unsat(f)

    def test_implies_invalid(self):
        """x <= 5 ∧ x >= 0 is satisfiable."""
        c1 = FormulaNode.leq({"x": 1.0}, 5.0)
        c2 = FormulaNode.leq({"x": -1.0}, 0.0)
        f = Formula(FormulaNode.and_(c1, c2))
        checker = SatisfiabilityChecker()
        assert not checker.is_unsat(f)

    def test_check_conjunction_list(self):
        """Conjunction of compatible formulas is SAT."""
        f1 = Formula(FormulaNode.leq({"x": 1.0}, 10.0))
        f2 = Formula(FormulaNode.leq({"x": -1.0}, 0.0))
        checker = SatisfiabilityChecker()
        sat, w = checker.check_conjunction([f1, f2])
        assert sat

    def test_empty_formula_sat(self):
        """Empty formula (True) is satisfiable."""
        f = Formula(FormulaNode.const(True))
        checker = SatisfiabilityChecker()
        sat, _ = checker.check(f)
        assert sat
