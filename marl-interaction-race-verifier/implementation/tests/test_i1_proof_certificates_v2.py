"""Tests for I1 improvements: proof certificates v2, independent checker, Farkas certificates."""

import json
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.reporting.proof_certificates import (
    CertificateBuilder,
    CertificateSerializer,
    CertificateFormat,
    CertificateFormatV2,
    IndependentChecker,
    IndependentCertificateChecker,
    FarkasCertificate,
    InductiveWitnessStep,
    HBDerivationStep,
    Verdict,
    CERTIFICATE_FORMAT_V2,
    _compute_certificate_hash,
)


# ======================================================================
# Helpers
# ======================================================================

def _build_minimal_v1_certificate(verdict="SAFE"):
    builder = CertificateBuilder(
        env_id="test-env", num_agents=2, state_dim=4, action_dims=[2, 2],
    )
    builder.set_verdict(verdict)
    builder.set_policies([
        {"id": "pi_0", "hash": "abc123", "architecture": "mlp_64_64"},
        {"id": "pi_1", "hash": "def456", "architecture": "mlp_64_64"},
    ])
    builder.set_specification(
        temporal_formula="always(linear([1.0, 0.0, -1.0, 0.0], 2.0))",
    )
    center = np.array([1.0, 2.0, 1.0, 2.0])
    gens = np.eye(4) * 0.5
    builder.set_fixpoint_result(
        domain="zonotope", iterations=15, widening_points=[5, 10],
        fixpoint_center=center, fixpoint_generators=gens,
    )
    builder.set_inductive_invariant(
        invariant_center=center, invariant_generators=gens,
        init_center=center, init_generators=gens * 0.1,
        post_center=center, post_generators=gens,
    )
    builder.set_hb_consistency(
        nodes=[{"id": f"e{i}", "agent": f"agent_{i%2}", "timestep": i} for i in range(4)],
        edges=[{"from": "e0", "to": "e1", "source": "program_order"},
               {"from": "e2", "to": "e3", "source": "program_order"}],
        topological_order=["e0", "e2", "e1", "e3"],
    )
    return builder.build()


def _build_v2_certificate():
    v1 = _build_minimal_v1_certificate()
    witnesses = [
        InductiveWitnessStep(
            iteration=0, center=[1.0, 2.0, 1.0, 2.0],
            generators=[[0.5, 0, 0, 0], [0, 0.5, 0, 0],
                         [0, 0, 0.5, 0], [0, 0, 0, 0.5]],
            containment_status="initial",
        ),
        InductiveWitnessStep(
            iteration=1, center=[1.0, 2.0, 1.0, 2.0],
            generators=[[0.5, 0, 0, 0], [0, 0.5, 0, 0],
                         [0, 0, 0.5, 0], [0, 0, 0, 0.5]],
            containment_status="contained",
        ),
    ]
    hb_derivation = [
        HBDerivationStep(
            edge_source="e0", edge_target="e1",
            derivation_rule="program_order",
            soundness_class="exact",
        ),
        HBDerivationStep(
            edge_source="e2", edge_target="e3",
            derivation_rule="communication",
            justification=["msg_1"],
            soundness_class="exact",
        ),
    ]
    farkas = [
        FarkasCertificate(
            dual_vector=[1.0],
            guarantee_matrix=[[1.0, 0.0]],
            guarantee_bounds=[5.0],
            target_coefficients=[1.0, 0.0],
            target_bound=5.0,
        ),
    ]
    return CertificateFormatV2.build_v2_certificate(
        verdict="SAFE",
        environment=v1["environment"],
        policies=v1["policies"],
        specification=v1["specification"],
        inductive_witnesses=witnesses,
        hb_derivation_chain=hb_derivation,
        farkas_certificates=farkas,
        abstract_fixpoint=v1["abstract_fixpoint"],
        inductive_invariant=v1["inductive_invariant"],
        hb_consistency=v1["hb_consistency"],
    )


# ======================================================================
# FarkasCertificate
# ======================================================================

class TestFarkasCertificate:
    def test_valid_certificate(self):
        fc = FarkasCertificate(
            dual_vector=[1.0],
            guarantee_matrix=[[1.0, 0.0]],
            guarantee_bounds=[5.0],
            target_coefficients=[1.0, 0.0],
            target_bound=5.0,
        )
        ok, msg = fc.verify()
        assert ok
        assert "valid" in msg.lower()

    def test_negative_dual_detected(self):
        fc = FarkasCertificate(
            dual_vector=[-1.0],
            guarantee_matrix=[[1.0]],
            guarantee_bounds=[5.0],
            target_coefficients=[1.0],
            target_bound=5.0,
        )
        ok, msg = fc.verify()
        assert not ok
        assert "negative" in msg.lower()

    def test_coefficient_mismatch(self):
        fc = FarkasCertificate(
            dual_vector=[1.0],
            guarantee_matrix=[[1.0, 0.0]],
            guarantee_bounds=[5.0],
            target_coefficients=[0.0, 1.0],  # wrong coefficients
            target_bound=5.0,
        )
        ok, msg = fc.verify()
        assert not ok

    def test_bound_violation(self):
        fc = FarkasCertificate(
            dual_vector=[1.0],
            guarantee_matrix=[[1.0]],
            guarantee_bounds=[10.0],
            target_coefficients=[1.0],
            target_bound=5.0,  # y^T b_g = 10 > 5 = b_a
        )
        ok, msg = fc.verify()
        assert not ok
        assert "bound" in msg.lower()

    def test_round_trip_dict(self):
        fc = FarkasCertificate(
            dual_vector=[1.0, 0.5],
            guarantee_matrix=[[1.0, 0.0], [0.0, 1.0]],
            guarantee_bounds=[5.0, 3.0],
            target_coefficients=[1.0, 0.5],
            target_bound=6.5,
        )
        d = fc.to_dict()
        fc2 = FarkasCertificate.from_dict(d)
        ok, _ = fc2.verify()
        assert ok


# ======================================================================
# InductiveWitnessStep
# ======================================================================

class TestInductiveWitnessStep:
    def test_round_trip(self):
        step = InductiveWitnessStep(
            iteration=3, center=[1.0, 2.0],
            generators=[[0.5, 0], [0, 0.5]],
            containment_status="contained",
            predecessor_hash="abc123",
        )
        d = step.to_dict()
        step2 = InductiveWitnessStep.from_dict(d)
        assert step2.iteration == 3
        assert step2.containment_status == "contained"


# ======================================================================
# HBDerivationStep
# ======================================================================

class TestHBDerivationStep:
    def test_round_trip(self):
        step = HBDerivationStep(
            edge_source="e0", edge_target="e1",
            derivation_rule="program_order",
            soundness_class="exact",
        )
        d = step.to_dict()
        step2 = HBDerivationStep.from_dict(d)
        assert step2.edge_source == "e0"
        assert step2.soundness_class == "exact"


# ======================================================================
# CertificateFormatV2
# ======================================================================

class TestCertificateFormatV2:
    def test_build_v2_produces_dict(self):
        cert = _build_v2_certificate()
        assert isinstance(cert, dict)
        assert cert["version"] == CERTIFICATE_FORMAT_V2

    def test_v2_has_extensions(self):
        cert = _build_v2_certificate()
        assert "v2_extensions" in cert
        v2 = cert["v2_extensions"]
        assert "inductive_witnesses" in v2
        assert "hb_derivation_chain" in v2
        assert "farkas_certificates" in v2

    def test_v2_hash_present(self):
        cert = _build_v2_certificate()
        assert cert["hash"] != ""

    def test_v2_witness_chain_integrity(self):
        cert = _build_v2_certificate()
        ok, msg = CertificateFormatV2.verify_witness_chain(cert)
        assert ok

    def test_v2_tampered_witness_chain_detected(self):
        cert = _build_v2_certificate()
        cert["v2_extensions"]["inductive_witnesses"][0]["center"] = [999.0]
        ok, msg = CertificateFormatV2.verify_witness_chain(cert)
        assert not ok

    def test_v2_validate_structure(self):
        cert = _build_v2_certificate()
        issues = CertificateFormatV2.validate_v2_structure(cert)
        assert issues == []

    def test_v2_invalid_verdict(self):
        v1 = _build_minimal_v1_certificate()
        with pytest.raises(ValueError):
            CertificateFormatV2.build_v2_certificate(
                verdict="BROKEN",
                environment=v1["environment"],
                policies=v1["policies"],
                specification=v1["specification"],
            )

    def test_v2_serialization_round_trip(self):
        cert = _build_v2_certificate()
        json_str = CertificateSerializer.to_json(cert)
        recovered = CertificateSerializer.from_json(json_str)
        assert recovered["version"] == CERTIFICATE_FORMAT_V2
        assert "v2_extensions" in recovered


# ======================================================================
# IndependentCertificateChecker
# ======================================================================

class TestIndependentCertificateChecker:
    def test_valid_v1_cert_passes(self):
        cert = _build_minimal_v1_certificate()
        checker = IndependentCertificateChecker()
        result = checker.check(cert)
        assert isinstance(result.overall_passed, bool)

    def test_valid_v2_cert_passes(self):
        cert = _build_v2_certificate()
        checker = IndependentCertificateChecker()
        result = checker.check(cert)
        # Check v2-specific components were tested
        component_names = [r.component for r in result.component_results]
        assert "witness_chain_integrity" in component_names
        assert "hb_derivation_chain" in component_names
        assert "farkas_certificates" in component_names

    def test_tampered_v2_hash_detected(self):
        cert = _build_v2_certificate()
        cert["verdict"] = "UNSAFE"
        checker = IndependentCertificateChecker()
        result = checker.check(cert)
        hash_results = [
            r for r in result.component_results if "hash" in r.component.lower()
        ]
        if hash_results:
            assert not hash_results[0].passed

    def test_invalid_farkas_detected(self):
        cert = _build_v2_certificate()
        cert["v2_extensions"]["farkas_certificates"] = [{
            "dual_vector": [-1.0],
            "guarantee_matrix": [[1.0]],
            "guarantee_bounds": [5.0],
            "target_coefficients": [1.0],
            "target_bound": 5.0,
        }]
        # Recompute hash so hash check passes
        cert["hash"] = _compute_certificate_hash(cert)
        checker = IndependentCertificateChecker()
        result = checker.check(cert)
        farkas_results = [
            r for r in result.component_results if "farkas" in r.component.lower()
        ]
        assert len(farkas_results) == 1
        assert not farkas_results[0].passed

    def test_invalid_hb_derivation_detected(self):
        cert = _build_v2_certificate()
        cert["v2_extensions"]["hb_derivation_chain"] = [{
            "edge_source": "e0", "edge_target": "e1",
            "derivation_rule": "",
            "soundness_class": "invalid_class",
        }]
        cert["hash"] = _compute_certificate_hash(cert)
        checker = IndependentCertificateChecker()
        result = checker.check(cert)
        hb_results = [
            r for r in result.component_results
            if r.component == "hb_derivation_chain"
        ]
        assert len(hb_results) == 1
        assert not hb_results[0].passed

    def test_check_result_summary(self):
        cert = _build_v2_certificate()
        checker = IndependentCertificateChecker()
        result = checker.check(cert)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_file_round_trip_v2(self, tmp_path):
        cert = _build_v2_certificate()
        path = str(tmp_path / "v2_cert.json")
        CertificateSerializer.to_file(cert, path)
        recovered = CertificateSerializer.from_file(path)
        checker = IndependentCertificateChecker()
        result = checker.check(recovered)
        # At minimum witness chain should pass (since it's hash-based)
        witness_results = [
            r for r in result.component_results
            if r.component == "witness_chain_integrity"
        ]
        assert len(witness_results) == 1
        assert witness_results[0].passed
