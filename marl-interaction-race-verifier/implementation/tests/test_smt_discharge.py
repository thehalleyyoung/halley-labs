"""Tests for marace.decomposition.smt_discharge — LP-based contract discharge."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.decomposition.contracts import (
    ConjunctivePredicate,
    LinearContract,
    LinearPredicate,
)
from marace.decomposition.smt_discharge import (
    SMTEncoder,
    SMTTheory,
    LPDischarger,
    ContractDischarger,
    CompositionSoundnessProver,
    FarkasCertificate,
    DischargeResult,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_pred(coeffs: dict, bound: float) -> LinearPredicate:
    return LinearPredicate(coefficients=coeffs, bound=bound)


def _make_conj(*preds) -> ConjunctivePredicate:
    cp = ConjunctivePredicate()
    for p in preds:
        cp.add(p)
    return cp


def _make_contract(name, assumptions, guarantees) -> LinearContract:
    return LinearContract(
        name=name,
        assumption=assumptions,
        guarantee=guarantees,
    )


# ======================================================================
# SMTEncoder
# ======================================================================

class TestSMTEncoder:
    """Test SMTEncoder encodes linear predicates correctly."""

    def test_encode_linear_predicate(self):
        """Encode a simple conjunctive predicate as Ax <= b."""
        pred = _make_conj(
            _make_pred({"x": 1.0, "y": 0.0}, 5.0),
            _make_pred({"x": 0.0, "y": 1.0}, 3.0),
        )
        encoder = SMTEncoder(theory=SMTTheory.QF_LRA)
        A, b = encoder.encode_linear_predicate(pred, ["x", "y"])
        assert A.shape == (2, 2)
        assert b.shape == (2,)
        np.testing.assert_allclose(A[0], [1.0, 0.0])
        np.testing.assert_allclose(A[1], [0.0, 1.0])
        np.testing.assert_allclose(b, [5.0, 3.0])

    def test_default_theory_is_qf_lra(self):
        encoder = SMTEncoder()
        assert encoder.theory == SMTTheory.QF_LRA

    def test_unsupported_theory_raises(self):
        with pytest.raises(NotImplementedError):
            SMTEncoder(theory=SMTTheory.QF_NRA)

    def test_check_feasibility_feasible(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([5.0, 3.0])
        feasible, witness = SMTEncoder.check_feasibility(A, b)
        assert feasible
        assert witness is not None
        assert np.all(A @ witness <= b + 1e-6)

    def test_check_feasibility_infeasible(self):
        # x <= -1 and x >= 2 => infeasible
        A = np.array([[1.0], [-1.0]])
        b = np.array([-2.0, -3.0])  # x <= -2 and -x <= -3 => x >= 3
        # Actually x <= -2 AND x >= 3 is infeasible
        feasible, witness = SMTEncoder.check_feasibility(A, b)
        assert not feasible

    def test_encode_contract_implication(self):
        premise = _make_conj(
            _make_pred({"x": 1.0}, 3.0),  # x <= 3
        )
        target = _make_conj(
            _make_pred({"x": 1.0}, 5.0),  # x <= 5
        )
        encoder = SMTEncoder()
        problems = encoder.encode_contract_implication(premise, target, ["x"])
        assert len(problems) == 1
        A, b_vec, c_j, d_j = problems[0]
        assert d_j == 5.0


# ======================================================================
# LPDischarger
# ======================================================================

class TestLPDischarger:
    """Test LPDischarger for contract assumption discharge."""

    def test_simple_implication_holds(self):
        """G: x <= 3 should imply A: x <= 5."""
        guarantee = _make_conj(_make_pred({"x": 1.0}, 3.0))
        assumption = _make_conj(_make_pred({"x": 1.0}, 5.0))

        discharger = LPDischarger()
        result = discharger.discharge_assumption(assumption, [guarantee])
        assert result.satisfied

    def test_implication_fails_with_counterexample(self):
        """G: x <= 5 should NOT imply A: x <= 3."""
        guarantee = _make_conj(_make_pred({"x": 1.0}, 5.0))
        assumption = _make_conj(_make_pred({"x": 1.0}, 3.0))

        discharger = LPDischarger()
        result = discharger.discharge_assumption(assumption, [guarantee])
        assert not result.satisfied

    def test_mutual_discharge_succeeds(self):
        """Two contracts where G1 implies A2 and G2 implies A1."""
        # Contract 1: assume y <= 4, guarantee x <= 3
        # Contract 2: assume x <= 5, guarantee y <= 2
        # G1 (x<=3) => A2 (x<=5) ✓
        # G2 (y<=2) => A1 (y<=4) ✓
        g1 = _make_conj(_make_pred({"x": 1.0}, 3.0))
        a1 = _make_conj(_make_pred({"y": 1.0}, 4.0))
        g2 = _make_conj(_make_pred({"y": 1.0}, 2.0))
        a2 = _make_conj(_make_pred({"x": 1.0}, 5.0))

        discharger = LPDischarger()
        # Discharge A1 against G2
        r1 = discharger.discharge_assumption(a1, [g2])
        assert r1.satisfied
        # Discharge A2 against G1
        r2 = discharger.discharge_assumption(a2, [g1])
        assert r2.satisfied

    def test_farkas_certificate_verifies(self):
        """When discharge succeeds, the Farkas certificate should verify."""
        guarantee = _make_conj(
            _make_pred({"x": 1.0}, 3.0),
            _make_pred({"y": 1.0}, 2.0),
        )
        assumption = _make_conj(_make_pred({"x": 1.0}, 5.0))

        discharger = LPDischarger()
        result = discharger.discharge_assumption(assumption, [guarantee])
        assert result.satisfied
        if result.witness is not None:
            assert result.witness.verify()

    def test_check_implication_convenience(self):
        """Test check_implication as synonym for discharge."""
        premise = _make_conj(_make_pred({"x": 1.0}, 3.0))
        target = _make_conj(_make_pred({"x": 1.0}, 5.0))

        discharger = LPDischarger()
        result = discharger.check_implication(premise, target)
        assert result.satisfied

    def test_multidimensional_discharge(self):
        """Discharge with multiple variables."""
        guarantee = _make_conj(
            _make_pred({"x": 1.0, "y": 0.0}, 2.0),
            _make_pred({"x": 0.0, "y": 1.0}, 3.0),
        )
        assumption = _make_conj(
            _make_pred({"x": 1.0, "y": 1.0}, 6.0),  # x+y <= 6
        )
        discharger = LPDischarger()
        result = discharger.discharge_assumption(assumption, [guarantee])
        # max x+y s.t. x<=2,y<=3 = 5 <= 6 ✓
        assert result.satisfied


# ======================================================================
# ContractDischarger
# ======================================================================

class TestContractDischarger:
    """Test ContractDischarger with AG semantics."""

    def test_composition_soundness_succeeds(self):
        """Verify AG composition rule when all assumptions are discharged."""
        c1 = _make_contract(
            "group1",
            assumptions=_make_conj(_make_pred({"y": 1.0}, 4.0)),
            guarantees=_make_conj(_make_pred({"x": 1.0}, 3.0)),
        )
        c2 = _make_contract(
            "group2",
            assumptions=_make_conj(_make_pred({"x": 1.0}, 5.0)),
            guarantees=_make_conj(_make_pred({"y": 1.0}, 2.0)),
        )
        discharger = ContractDischarger()
        result = discharger.verify_composition_soundness({"g1": c1, "g2": c2})
        assert result.sound

    def test_composition_soundness_fails(self):
        """Verify failure when assumptions cannot be discharged."""
        c1 = _make_contract(
            "group1",
            assumptions=_make_conj(_make_pred({"y": 1.0}, 1.0)),
            guarantees=_make_conj(_make_pred({"x": 1.0}, 3.0)),
        )
        c2 = _make_contract(
            "group2",
            assumptions=_make_conj(_make_pred({"x": 1.0}, 2.0)),
            guarantees=_make_conj(_make_pred({"y": 1.0}, 5.0)),
        )
        discharger = ContractDischarger()
        result = discharger.verify_composition_soundness({"g1": c1, "g2": c2})
        # G2 (y<=5) does NOT imply A1 (y<=1), so unsound
        assert not result.sound
        assert len(result.undischarged) > 0

    def test_counterexample_on_failure(self):
        """When discharge fails, a counterexample should be provided."""
        g = _make_conj(_make_pred({"x": 1.0}, 10.0))
        a = _make_conj(_make_pred({"x": 1.0}, 3.0))
        c_other = _make_contract("other", _make_conj(), g)

        discharger = ContractDischarger()
        result = discharger.discharge_assumption(a, [c_other])
        assert not result.satisfied


# ======================================================================
# CompositionSoundnessProver
# ======================================================================

class TestCompositionSoundnessProver:
    """Test CompositionSoundnessProver generates valid proof chains."""

    def test_prove_generates_chain(self):
        """Prove should return a CompositionProof with steps."""
        c1 = _make_contract(
            "g1",
            assumptions=_make_conj(_make_pred({"y": 1.0}, 4.0)),
            guarantees=_make_conj(_make_pred({"x": 1.0}, 2.0)),
        )
        c2 = _make_contract(
            "g2",
            assumptions=_make_conj(_make_pred({"x": 1.0}, 5.0)),
            guarantees=_make_conj(_make_pred({"y": 1.0}, 1.0)),
        )
        prover = CompositionSoundnessProver()
        proof = prover.prove({"g1": c1, "g2": c2})
        assert proof.is_complete
        assert proof.num_steps > 0
        assert proof.num_undischarged == 0

    def test_proof_verifies_independently(self):
        """The proof should pass independent verification."""
        c1 = _make_contract(
            "g1",
            assumptions=_make_conj(_make_pred({"y": 1.0}, 10.0)),
            guarantees=_make_conj(_make_pred({"x": 1.0}, 2.0)),
        )
        c2 = _make_contract(
            "g2",
            assumptions=_make_conj(_make_pred({"x": 1.0}, 3.0)),
            guarantees=_make_conj(_make_pred({"y": 1.0}, 5.0)),
        )
        prover = CompositionSoundnessProver()
        proof = prover.prove({"g1": c1, "g2": c2})
        assert proof.is_complete
        assert proof.verify_independently()

    def test_incomplete_proof_summary(self):
        """Incomplete proof should report undischarged clauses."""
        c1 = _make_contract(
            "g1",
            assumptions=_make_conj(_make_pred({"y": 1.0}, 1.0)),
            guarantees=_make_conj(_make_pred({"x": 1.0}, 3.0)),
        )
        c2 = _make_contract(
            "g2",
            assumptions=_make_conj(_make_pred({"x": 1.0}, 2.0)),
            guarantees=_make_conj(_make_pred({"y": 1.0}, 5.0)),
        )
        prover = CompositionSoundnessProver()
        proof = prover.prove({"g1": c1, "g2": c2})
        assert not proof.is_complete
        summary = proof.summary()
        assert "INCOMPLETE" in summary

    def test_proof_elapsed_time(self):
        c1 = _make_contract(
            "g1",
            assumptions=_make_conj(_make_pred({"y": 1.0}, 10.0)),
            guarantees=_make_conj(_make_pred({"x": 1.0}, 2.0)),
        )
        c2 = _make_contract(
            "g2",
            assumptions=_make_conj(_make_pred({"x": 1.0}, 5.0)),
            guarantees=_make_conj(_make_pred({"y": 1.0}, 3.0)),
        )
        prover = CompositionSoundnessProver()
        proof = prover.prove({"g1": c1, "g2": c2})
        assert proof.elapsed_seconds >= 0


# ======================================================================
# FarkasCertificate
# ======================================================================

class TestFarkasCertificate:
    """Test FarkasCertificate independent verification."""

    def test_valid_certificate(self):
        """A correctly constructed certificate should verify."""
        # Premise: x <= 3 (A = [[1]], b = [3])
        # Target: x <= 5 (c = [1], d = 5)
        # λ = [1] => A^T λ = [1] = c, b^T λ = 3 <= 5 ✓
        cert = FarkasCertificate(
            dual_multipliers=np.array([1.0]),
            premise_matrix=np.array([[1.0]]),
            premise_rhs=np.array([3.0]),
            target_normal=np.array([1.0]),
            target_bound=5.0,
        )
        assert cert.verify()

    def test_invalid_certificate(self):
        """A certificate with wrong multipliers should fail."""
        cert = FarkasCertificate(
            dual_multipliers=np.array([1.0]),
            premise_matrix=np.array([[1.0]]),
            premise_rhs=np.array([10.0]),
            target_normal=np.array([1.0]),
            target_bound=5.0,  # b^T λ = 10 > 5
        )
        assert not cert.verify()

    def test_negative_multiplier_fails(self):
        cert = FarkasCertificate(
            dual_multipliers=np.array([-1.0]),
            premise_matrix=np.array([[1.0]]),
            premise_rhs=np.array([3.0]),
            target_normal=np.array([1.0]),
            target_bound=5.0,
        )
        assert not cert.verify()
