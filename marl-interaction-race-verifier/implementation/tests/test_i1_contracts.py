"""Tests for I1 improvements: contract composition soundness, refinement, weakening/strengthening."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.decomposition.contracts import (
    LinearContract,
    LinearPredicate,
    ConjunctivePredicate,
    ContractComposition,
    CompositionSoundnessTheorem,
    ContractRefinementChecker,
    ContractWeakeningStrengthening,
    ProofObligation,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_contract(name, assume_clauses, guarantee_clauses):
    assume = ConjunctivePredicate()
    for coeffs, bound in assume_clauses:
        assume.add(LinearPredicate(coeffs, bound))
    guarantee = ConjunctivePredicate()
    for coeffs, bound in guarantee_clauses:
        guarantee.add(LinearPredicate(coeffs, bound))
    return LinearContract(name=name, assumption=assume, guarantee=guarantee)


# ======================================================================
# CompositionSoundnessTheorem
# ======================================================================

class TestCompositionSoundnessTheorem:
    def test_two_contracts_mutual_discharge(self):
        """C1 assumes x <= 5 (guaranteed by C2), C2 assumes y <= 3 (guaranteed by C1).
        This is circular reasoning and should be detected as unsound."""
        c1 = _make_contract(
            "c1",
            assume_clauses=[( {"x": 1.0}, 5.0 )],
            guarantee_clauses=[( {"y": 1.0}, 3.0 )],
        )
        c2 = _make_contract(
            "c2",
            assume_clauses=[( {"y": 1.0}, 3.0 )],
            guarantee_clauses=[( {"x": 1.0}, 5.0 )],
        )
        theorem = CompositionSoundnessTheorem([c1, c2])
        is_sound, obligations = theorem.check_soundness()
        # Circular discharge is unsound
        assert not is_sound
        non_circ = [ob for ob in obligations if ob.name == "non_circularity"]
        assert len(non_circ) == 1
        assert not non_circ[0].verified

    def test_acyclic_discharge_is_sound(self):
        """C1 has no assumptions and guarantees x. C2 assumes x (discharged by C1)."""
        c1 = _make_contract(
            "c1",
            assume_clauses=[],
            guarantee_clauses=[( {"x": 1.0}, 5.0 )],
        )
        c2 = _make_contract(
            "c2",
            assume_clauses=[( {"x": 1.0}, 5.0 )],
            guarantee_clauses=[( {"y": 1.0}, 3.0 )],
        )
        theorem = CompositionSoundnessTheorem([c1, c2])
        is_sound, obligations = theorem.check_soundness()
        assert is_sound
        assert all(ob.verified for ob in obligations)

    def test_undischarged_assumption_fails(self):
        """C1 assumes z <= 10, but no contract guarantees z."""
        c1 = _make_contract(
            "c1",
            assume_clauses=[( {"z": 1.0}, 10.0 )],
            guarantee_clauses=[( {"y": 1.0}, 3.0 )],
        )
        c2 = _make_contract(
            "c2",
            assume_clauses=[],
            guarantee_clauses=[( {"x": 1.0}, 5.0 )],
        )
        theorem = CompositionSoundnessTheorem([c1, c2])
        is_sound, obligations = theorem.check_soundness()
        assert not is_sound

    def test_empty_contracts_is_sound(self):
        c1 = _make_contract("c1", [], [( {"x": 1.0}, 5.0 )])
        c2 = _make_contract("c2", [], [( {"y": 1.0}, 3.0 )])
        theorem = CompositionSoundnessTheorem([c1, c2])
        is_sound, obligations = theorem.check_soundness()
        assert is_sound

    def test_non_circularity_check(self):
        """Acyclic discharge graph should pass."""
        c1 = _make_contract(
            "c1",
            assume_clauses=[],
            guarantee_clauses=[( {"x": 1.0}, 5.0 )],
        )
        c2 = _make_contract(
            "c2",
            assume_clauses=[( {"x": 1.0}, 5.0 )],
            guarantee_clauses=[( {"y": 1.0}, 3.0 )],
        )
        theorem = CompositionSoundnessTheorem([c1, c2])
        is_sound, obligations = theorem.check_soundness()
        assert is_sound
        non_circ = [ob for ob in obligations if ob.name == "non_circularity"]
        assert len(non_circ) == 1
        assert non_circ[0].verified

    def test_generate_obligations_count(self):
        c1 = _make_contract(
            "c1",
            assume_clauses=[( {"x": 1.0}, 5.0 ), ( {"y": 1.0}, 3.0 )],
            guarantee_clauses=[( {"z": 1.0}, 10.0 )],
        )
        c2 = _make_contract(
            "c2",
            assume_clauses=[( {"z": 1.0}, 10.0 )],
            guarantee_clauses=[( {"x": 1.0}, 5.0 ), ( {"y": 1.0}, 3.0 )],
        )
        theorem = CompositionSoundnessTheorem([c1, c2])
        obligations = theorem.generate_obligations()
        # 2 assumptions from c1 + 1 from c2 + 1 non-circularity = 4
        assert len(obligations) == 4

    def test_is_sound_none_before_check(self):
        c1 = _make_contract("c1", [], [])
        theorem = CompositionSoundnessTheorem([c1])
        assert theorem.is_sound is None


# ======================================================================
# ContractRefinementChecker
# ======================================================================

class TestContractRefinementChecker:
    def test_identical_contract_refines_itself(self):
        c = _make_contract(
            "c",
            assume_clauses=[( {"x": 1.0}, 5.0 )],
            guarantee_clauses=[( {"y": 1.0}, 3.0 )],
        )
        checker = ContractRefinementChecker()
        refines, obligations = checker.check_refinement(c, c)
        assert refines

    def test_stronger_guarantee_refines(self):
        c1 = _make_contract(
            "c1",
            assume_clauses=[( {"x": 1.0}, 5.0 )],
            guarantee_clauses=[( {"y": 1.0}, 2.0 )],  # tighter bound
        )
        c2 = _make_contract(
            "c2",
            assume_clauses=[( {"x": 1.0}, 5.0 )],
            guarantee_clauses=[( {"y": 1.0}, 3.0 )],
        )
        checker = ContractRefinementChecker()
        refines, _ = checker.check_refinement(c1, c2)
        assert refines

    def test_weaker_guarantee_does_not_refine(self):
        c1 = _make_contract(
            "c1",
            assume_clauses=[( {"x": 1.0}, 5.0 )],
            guarantee_clauses=[( {"y": 1.0}, 10.0 )],  # weaker
        )
        c2 = _make_contract(
            "c2",
            assume_clauses=[( {"x": 1.0}, 5.0 )],
            guarantee_clauses=[( {"y": 1.0}, 3.0 )],
        )
        checker = ContractRefinementChecker()
        refines, _ = checker.check_refinement(c1, c2)
        assert not refines


# ======================================================================
# ContractWeakeningStrengthening
# ======================================================================

class TestContractWeakeningStrengthening:
    def test_weaken_guarantee(self):
        c = _make_contract(
            "c",
            assume_clauses=[],
            guarantee_clauses=[( {"x": 1.0}, 5.0 )],
        )
        ws = ContractWeakeningStrengthening()
        weakened, obligations = ws.weaken_guarantee(c, factor=1.5)
        assert weakened.name == "c_weakened_g"
        # Weakened bound should be 5.0 * 1.5 = 7.5
        assert weakened.guarantee.clauses[0].bound == pytest.approx(7.5)
        assert len(obligations) == 1

    def test_strengthen_guarantee(self):
        c = _make_contract(
            "c",
            assume_clauses=[],
            guarantee_clauses=[( {"x": 1.0}, 5.0 )],
        )
        ws = ContractWeakeningStrengthening()
        strengthened, obligations = ws.strengthen_guarantee(c, factor=0.8)
        assert strengthened.name == "c_strengthened_g"
        assert strengthened.guarantee.clauses[0].bound == pytest.approx(4.0)
        assert len(obligations) == 1

    def test_weaken_assumption(self):
        c = _make_contract(
            "c",
            assume_clauses=[( {"x": 1.0}, 5.0 )],
            guarantee_clauses=[( {"y": 1.0}, 3.0 )],
        )
        ws = ContractWeakeningStrengthening()
        weakened, obligations = ws.weaken_assumption(c, factor=2.0)
        assert weakened.name == "c_weakened_a"
        assert weakened.assumption.clauses[0].bound == pytest.approx(10.0)
        # Assumption weakening is always sound
        assert obligations[0].verified

    def test_weakened_guarantee_is_weaker(self):
        """Weakened contract should satisfy more assignments."""
        c = _make_contract(
            "c",
            assume_clauses=[],
            guarantee_clauses=[( {"x": 1.0}, 5.0 )],
        )
        ws = ContractWeakeningStrengthening()
        weakened, _ = ws.weaken_guarantee(c, factor=2.0)
        # x = 8 violates original but satisfies weakened
        assert not c.guarantee.evaluate({"x": 8.0})
        assert weakened.guarantee.evaluate({"x": 8.0})

    def test_strengthened_guarantee_is_stronger(self):
        """Strengthened contract should satisfy fewer assignments."""
        c = _make_contract(
            "c",
            assume_clauses=[],
            guarantee_clauses=[( {"x": 1.0}, 10.0 )],
        )
        ws = ContractWeakeningStrengthening()
        strengthened, _ = ws.strengthen_guarantee(c, factor=0.5)
        # x = 8 satisfies original but violates strengthened
        assert c.guarantee.evaluate({"x": 8.0})
        assert not strengthened.guarantee.evaluate({"x": 8.0})


# ======================================================================
# ProofObligation
# ======================================================================

class TestProofObligation:
    def test_create(self):
        ob = ProofObligation(
            name="test_ob",
            description="Test obligation",
            verified=True,
            evidence={"key": "value"},
        )
        assert ob.name == "test_ob"
        assert ob.verified
        assert ob.evidence == {"key": "value"}

    def test_defaults(self):
        ob = ProofObligation(name="ob", description="desc")
        assert not ob.verified
        assert ob.evidence is None


# ======================================================================
# Farkas certificate generation (requires scipy)
# ======================================================================

class TestFarkasCertificateGeneration:
    def test_generate_farkas_for_simple_entailment(self):
        """G: x <= 5, target: x <= 10. Should be trivially entailed."""
        try:
            from scipy.optimize import linprog  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        c = _make_contract(
            "c",
            assume_clauses=[],
            guarantee_clauses=[( {"x": 1.0}, 5.0 )],
        )
        target = LinearPredicate({"x": 1.0}, 10.0)
        checker = ContractRefinementChecker()
        cert = checker.generate_farkas_certificate(c, target)
        assert cert is not None
        assert cert["dual_vector"][0] >= 0

    def test_infeasible_farkas(self):
        """G: x <= 5, target: -x <= -10 (x >= 10). Not entailed."""
        try:
            from scipy.optimize import linprog  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available")

        c = _make_contract(
            "c",
            assume_clauses=[],
            guarantee_clauses=[( {"x": 1.0}, 5.0 )],
        )
        target = LinearPredicate({"x": -1.0}, -10.0)
        checker = ContractRefinementChecker()
        cert = checker.generate_farkas_certificate(c, target)
        assert cert is None
