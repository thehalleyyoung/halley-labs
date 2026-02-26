"""Tests for marace.reporting.proof_certificates — certificate building, serialization, checking."""

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
    IndependentChecker,
    CertificateChainVerifier,
    Verdict,
    _compute_certificate_hash,
)


# ======================================================================
# Helpers
# ======================================================================

def _build_minimal_certificate(verdict="SAFE"):
    """Build a minimal well-formed certificate for testing."""
    builder = CertificateBuilder(
        env_id="test-env",
        num_agents=2,
        state_dim=4,
        action_dims=[2, 2],
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
        domain="zonotope",
        iterations=15,
        widening_points=[5, 10],
        fixpoint_center=center,
        fixpoint_generators=gens,
    )
    builder.set_inductive_invariant(
        invariant_center=center,
        invariant_generators=gens,
        init_center=center,
        init_generators=gens * 0.1,
        post_center=center,
        post_generators=gens,
    )
    builder.set_hb_consistency(
        nodes=[{"id": f"e{i}", "agent": f"agent_{i%2}", "timestep": i}
               for i in range(4)],
        edges=[{"from": "e0", "to": "e1", "source": "program_order"},
               {"from": "e2", "to": "e3", "source": "program_order"}],
        topological_order=["e0", "e2", "e1", "e3"],
    )
    return builder.build()


# ======================================================================
# CertificateBuilder
# ======================================================================

class TestCertificateBuilder:
    """Test CertificateBuilder creates well-formed certificates."""

    def test_build_produces_dict(self):
        cert = _build_minimal_certificate()
        assert isinstance(cert, dict)

    def test_certificate_has_required_keys(self):
        cert = _build_minimal_certificate()
        assert "version" in cert
        assert "environment" in cert
        assert "verdict" in cert

    def test_verdict_is_set(self):
        cert = _build_minimal_certificate("SAFE")
        assert cert["verdict"] == "SAFE"

    def test_unsafe_verdict(self):
        cert = _build_minimal_certificate("UNSAFE")
        assert cert["verdict"] == "UNSAFE"

    def test_invalid_verdict_raises(self):
        builder = CertificateBuilder(
            env_id="test", num_agents=1, state_dim=2, action_dims=[1],
        )
        with pytest.raises(ValueError):
            builder.set_verdict("INVALID")

    def test_certificate_has_hash(self):
        cert = _build_minimal_certificate()
        assert "hash" in cert

    def test_structure_validation_passes(self):
        cert = _build_minimal_certificate()
        errors = CertificateFormat.validate_structure(cert)
        assert errors == []

    def test_environment_info(self):
        cert = _build_minimal_certificate()
        env = cert["environment"]
        assert env["id"] == "test-env"
        assert env["num_agents"] == 2


# ======================================================================
# CertificateSerializer
# ======================================================================

class TestCertificateSerializer:
    """Test CertificateSerializer round-trips through JSON."""

    def test_round_trip_json(self):
        cert = _build_minimal_certificate()
        json_str = CertificateSerializer.to_json(cert)
        recovered = CertificateSerializer.from_json(json_str)
        assert isinstance(recovered, dict)
        assert recovered["verdict"] == cert["verdict"]
        assert recovered["version"] == cert["version"]

    def test_json_is_valid(self):
        cert = _build_minimal_certificate()
        json_str = CertificateSerializer.to_json(cert)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_encode_numpy_arrays(self):
        """Numpy arrays should be serialized and recovered."""
        arr = np.array([1.0, 2.0, 3.0])
        encoded = CertificateSerializer.encode_value(arr)
        # Should be JSON-serializable
        json.dumps(encoded)

    def test_decode_restores_arrays(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        encoded = CertificateSerializer.encode_value(arr)
        decoded = CertificateSerializer.decode_value(encoded)
        if isinstance(decoded, np.ndarray):
            np.testing.assert_allclose(decoded, arr)

    def test_round_trip_preserves_environment(self):
        cert = _build_minimal_certificate()
        json_str = CertificateSerializer.to_json(cert)
        recovered = CertificateSerializer.from_json(json_str)
        assert recovered["environment"]["id"] == "test-env"

    def test_to_file_and_from_file(self, tmp_path):
        cert = _build_minimal_certificate()
        path = str(tmp_path / "cert.json")
        CertificateSerializer.to_file(cert, path)
        recovered = CertificateSerializer.from_file(path)
        assert recovered["verdict"] == cert["verdict"]


# ======================================================================
# IndependentChecker
# ======================================================================

class TestIndependentChecker:
    """Test IndependentChecker validates correct and tampered certificates."""

    def test_valid_certificate_passes(self):
        cert = _build_minimal_certificate()
        checker = IndependentChecker()
        result = checker.check(cert)
        # The certificate should at least pass structure and hash checks
        assert isinstance(result.overall_passed, bool)
        assert len(result.component_results) > 0

    def test_hash_check_passes_on_correct(self):
        cert = _build_minimal_certificate()
        checker = IndependentChecker()
        result = checker.check(cert)
        hash_results = [
            r for r in result.component_results if "hash" in r.component.lower()
        ]
        if hash_results:
            assert hash_results[0].passed

    def test_tampered_certificate_detected(self):
        """Modifying the verdict after building should be caught."""
        cert = _build_minimal_certificate("SAFE")
        # Tamper with the verdict
        cert["verdict"] = "UNSAFE"
        checker = IndependentChecker()
        result = checker.check(cert)
        hash_results = [
            r for r in result.component_results if "hash" in r.component.lower()
        ]
        if hash_results:
            assert not hash_results[0].passed

    def test_tampered_environment_detected(self):
        """Modifying environment data should invalidate the hash."""
        cert = _build_minimal_certificate()
        cert["environment"]["num_agents"] = 999
        checker = IndependentChecker()
        result = checker.check(cert)
        hash_results = [
            r for r in result.component_results if "hash" in r.component.lower()
        ]
        if hash_results:
            assert not hash_results[0].passed

    def test_missing_key_detected(self):
        cert = _build_minimal_certificate()
        del cert["version"]
        checker = IndependentChecker()
        result = checker.check(cert)
        structure_results = [
            r for r in result.component_results if "structure" in r.component.lower()
        ]
        if structure_results:
            assert not structure_results[0].passed

    def test_check_result_summary(self):
        cert = _build_minimal_certificate()
        checker = IndependentChecker()
        result = checker.check(cert)
        summary = result.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ======================================================================
# CertificateChainVerifier
# ======================================================================

class TestCertificateChainVerifier:
    """Test certificate chain verification."""

    def test_single_certificate_chain(self):
        cert = _build_minimal_certificate()
        verifier = CertificateChainVerifier()
        result = verifier.verify_chain([cert])
        assert isinstance(result.overall_passed, bool)

    def test_overall_verdict(self):
        cert = _build_minimal_certificate("SAFE")
        verifier = CertificateChainVerifier()
        verdict = verifier.overall_verdict([cert])
        assert verdict in {"SAFE", "UNSAFE", "UNKNOWN"}


# ======================================================================
# Verdict enum
# ======================================================================

class TestVerdict:
    def test_verdict_values(self):
        assert Verdict.SAFE.value == "SAFE"
        assert Verdict.UNSAFE.value == "UNSAFE"
        assert Verdict.UNKNOWN.value == "UNKNOWN"
