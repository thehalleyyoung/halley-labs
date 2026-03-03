"""
Comprehensive tests for proof certificate system.

Tests:
- ProofCertificate creation and serialization
- JSON export/import roundtrip
- Certificate verification (verify_certificate)
- Certificate chaining for composition
- Audit log functionality
"""

import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dp_forge.verification.proof_certificate import (
    ProofCertificate,
    CertificateBuilder,
    BoundDerivation,
    AuditEvent,
    MechanismSpec,
    BoundType,
    VerificationMethod,
    verify_certificate,
    load_certificate,
    save_certificate,
    compose_certificates,
    generate_certificate_report,
    CertificateChain,
    CertificateValidator,
    batch_verify_certificates,
    merge_certificates,
    export_certificate_to_latex,
)
from dp_forge.exceptions import VerificationError


class TestBoundDerivation:
    """Test BoundDerivation dataclass."""
    
    def test_bound_derivation_creation(self):
        bound = BoundDerivation(
            bound_type="PRIVACY_LOSS",
            pair_indices=(0, 1),
            bound_value=0.5,
            target_value=1.0,
            satisfies=True,
        )
        
        assert bound.bound_type == "PRIVACY_LOSS"
        assert bound.pair_indices == (0, 1)
        assert bound.satisfies is True
    
    def test_bound_derivation_to_dict(self):
        bound = BoundDerivation(
            bound_type="HOCKEY_STICK",
            pair_indices=(1, 2),
            bound_value=0.05,
            target_value=0.1,
            satisfies=True,
            computation_steps=["step1", "step2"],
        )
        
        d = bound.to_dict()
        
        assert d["bound_type"] == "HOCKEY_STICK"
        assert d["pair_indices"] == [1, 2]
        assert d["bound_value"] == 0.05
        assert len(d["computation_steps"]) == 2
    
    def test_bound_derivation_from_dict(self):
        d = {
            "bound_type": "PRIVACY_LOSS",
            "pair_indices": [0, 1],
            "bound_value": 0.5,
            "target_value": 1.0,
            "satisfies": True,
            "computation_steps": [],
            "method": "INTERVAL_ARITHMETIC",
        }
        
        bound = BoundDerivation.from_dict(d)
        
        assert bound.bound_type == "PRIVACY_LOSS"
        assert bound.pair_indices == (0, 1)


class TestAuditEvent:
    """Test AuditEvent dataclass."""
    
    def test_audit_event_creation(self):
        event = AuditEvent(
            timestamp="2024-01-01T00:00:00Z",
            event_type="VERIFICATION",
            description="Started verification",
        )
        
        assert event.timestamp == "2024-01-01T00:00:00Z"
        assert event.event_type == "VERIFICATION"
    
    def test_audit_event_to_dict(self):
        event = AuditEvent(
            timestamp="2024-01-01T00:00:00Z",
            event_type="VERIFICATION",
            description="Test event",
            metadata={"key": "value"},
        )
        
        d = event.to_dict()
        
        assert d["timestamp"] == "2024-01-01T00:00:00Z"
        assert d["metadata"]["key"] == "value"
    
    def test_audit_event_from_dict(self):
        d = {
            "timestamp": "2024-01-01T00:00:00Z",
            "event_type": "TEST",
            "description": "Test",
            "metadata": {},
        }
        
        event = AuditEvent.from_dict(d)
        
        assert event.event_type == "TEST"


class TestMechanismSpec:
    """Test MechanismSpec dataclass."""
    
    def test_mechanism_spec_creation(self):
        spec = MechanismSpec(
            n_databases=3,
            n_outputs=4,
            mechanism_hash="abc123",
            adjacency_relation=[(0, 1), (1, 2)],
        )
        
        assert spec.n_databases == 3
        assert spec.n_outputs == 4
        assert len(spec.adjacency_relation) == 2
    
    def test_mechanism_spec_to_dict(self):
        spec = MechanismSpec(
            n_databases=2,
            n_outputs=3,
            mechanism_hash="hash",
            adjacency_relation=[(0, 1)],
            metadata={"info": "test"},
        )
        
        d = spec.to_dict()
        
        assert d["n_databases"] == 2
        assert d["adjacency_relation"] == [[0, 1]]
        assert d["metadata"]["info"] == "test"
    
    def test_mechanism_spec_from_dict(self):
        d = {
            "n_databases": 2,
            "n_outputs": 3,
            "mechanism_hash": "hash",
            "adjacency_relation": [[0, 1], [1, 2]],
            "metadata": {},
        }
        
        spec = MechanismSpec.from_dict(d)
        
        assert spec.n_databases == 2
        assert spec.adjacency_relation == [(0, 1), (1, 2)]


class TestCertificateBuilder:
    """Test CertificateBuilder."""
    
    def test_builder_initialization(self):
        prob_table = np.array([[0.5, 0.5], [0.5, 0.5]])
        edges = [(0, 1)]
        
        builder = CertificateBuilder(
            prob_table, epsilon=1.0, delta=0.0, edges=edges
        )
        
        assert builder.epsilon == 1.0
        assert builder.delta == 0.0
        assert len(builder.edges) == 1
        assert len(builder.audit_log) > 0
    
    def test_builder_generates_unique_id(self):
        prob_table = np.array([[0.5, 0.5]])
        edges = []
        
        builder1 = CertificateBuilder(prob_table, 1.0, 0.0, edges)
        builder2 = CertificateBuilder(prob_table, 1.0, 0.0, edges)
        
        assert builder1.certificate_id != builder2.certificate_id
    
    def test_builder_computes_mechanism_hash(self):
        prob_table = np.array([[0.5, 0.5], [0.4, 0.6]])
        edges = [(0, 1)]
        
        builder = CertificateBuilder(prob_table, 1.0, 0.0, edges)
        hash_val = builder._compute_mechanism_hash()
        
        expected = hashlib.sha256(prob_table.tobytes()).hexdigest()
        assert hash_val == expected
    
    def test_builder_add_bound(self):
        prob_table = np.array([[0.5, 0.5]])
        edges = []
        
        builder = CertificateBuilder(prob_table, 1.0, 0.0, edges)
        builder.add_bound(
            pair=(0, 1),
            bound_value=0.5,
            target_value=1.0,
            satisfies=True,
        )
        
        assert len(builder.bound_derivations) == 1
        assert builder.bound_derivations[0].pair_indices == (0, 1)
    
    def test_builder_add_hockey_stick_bound(self):
        prob_table = np.array([[0.5, 0.5]])
        edges = []
        
        builder = CertificateBuilder(prob_table, 1.0, 0.1, edges)
        builder.add_hockey_stick_bound((0, 1), hs_value=0.05, satisfies=True)
        
        assert len(builder.bound_derivations) == 1
        assert builder.bound_derivations[0].bound_type == "HOCKEY_STICK"
    
    def test_builder_add_renyi_bound(self):
        prob_table = np.array([[0.5, 0.5]])
        edges = []
        
        builder = CertificateBuilder(prob_table, 1.0, 0.0, edges)
        builder.add_renyi_bound((0, 1), renyi_value=0.8, renyi_epsilon=1.0, satisfies=True)
        
        assert len(builder.bound_derivations) == 1
        assert builder.bound_derivations[0].bound_type == "RENYI_DIVERGENCE"
    
    def test_builder_build_valid_certificate(self):
        prob_table = np.array([[0.5, 0.5], [0.5, 0.5]])
        edges = [(0, 1)]
        
        builder = CertificateBuilder(prob_table, 1.0, 0.0, edges)
        builder.add_bound((0, 1), 0.5, 1.0, True)
        
        cert = builder.build(is_valid=True, confidence=1.0)
        
        assert isinstance(cert, ProofCertificate)
        assert cert.is_valid is True
        assert cert.confidence == 1.0
        assert len(cert.bound_derivations) == 1
    
    def test_builder_adds_audit_events(self):
        prob_table = np.array([[0.5, 0.5]])
        edges = []
        
        builder = CertificateBuilder(prob_table, 1.0, 0.0, edges)
        initial_events = len(builder.audit_log)
        
        cert = builder.build(is_valid=True)
        
        assert len(cert.audit_log) > initial_events


class TestProofCertificate:
    """Test ProofCertificate class."""
    
    def test_certificate_creation(self):
        spec = MechanismSpec(2, 3, "hash", [(0, 1)])
        cert = ProofCertificate(
            certificate_id="cert123",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="INTERVAL_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        assert cert.certificate_id == "cert123"
        assert cert.epsilon == 1.0
        assert cert.is_valid is True
    
    def test_certificate_to_dict(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        bound = BoundDerivation("PRIVACY_LOSS", (0, 1), 0.5, 1.0, True)
        
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[bound],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        d = cert.to_dict()
        
        assert d["certificate_id"] == "cert1"
        assert d["epsilon"] == 1.0
        assert len(d["bound_derivations"]) == 1
    
    def test_certificate_from_dict(self):
        d = {
            "certificate_id": "cert1",
            "mechanism_spec": {
                "n_databases": 2,
                "n_outputs": 2,
                "mechanism_hash": "hash",
                "adjacency_relation": [[0, 1]],
                "metadata": {},
            },
            "epsilon": 1.0,
            "delta": 0.0,
            "is_valid": True,
            "verification_method": "FLOAT_ARITHMETIC",
            "bound_derivations": [],
            "tolerance": 1e-9,
            "confidence": 1.0,
            "created_at": "2024-01-01T00:00:00Z",
            "verifier_version": "1.0.0",
            "audit_log": [],
            "parent_certificates": [],
            "signature": None,
        }
        
        cert = ProofCertificate.from_dict(d)
        
        assert cert.certificate_id == "cert1"
        assert cert.epsilon == 1.0
    
    def test_certificate_summary(self):
        spec = MechanismSpec(2, 3, "hash", [(0, 1)])
        cert = ProofCertificate(
            certificate_id="cert123",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.1,
            is_valid=True,
            verification_method="INTERVAL_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=0.95,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        summary = cert.summary()
        
        assert "VALID" in summary
        assert "cert123" in summary
        assert "ε=1.0" in summary
        assert "95.00%" in summary


class TestCertificateVerification:
    """Test certificate verification."""
    
    def test_verify_certificate_valid(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        bound = BoundDerivation("PRIVACY_LOSS", (0, 1), 0.5, 1.0, True)
        
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[bound],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        result = verify_certificate(cert)
        assert result is True
    
    def test_verify_certificate_checks_hash(self):
        prob_table = np.array([[0.5, 0.5], [0.5, 0.5]])
        hash_val = hashlib.sha256(prob_table.tobytes()).hexdigest()
        
        spec = MechanismSpec(2, 2, hash_val, [(0, 1)])
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        result = verify_certificate(cert, prob_table)
        assert result is True
    
    def test_verify_certificate_rejects_hash_mismatch(self):
        prob_table = np.array([[0.5, 0.5]])
        wrong_hash = "wrong_hash"
        
        spec = MechanismSpec(1, 2, wrong_hash, [])
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        with pytest.raises(VerificationError, match="hash mismatch"):
            verify_certificate(cert, prob_table)
    
    def test_verify_certificate_rejects_invalid_epsilon(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=-1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        with pytest.raises(VerificationError, match="Invalid epsilon"):
            verify_certificate(cert)
    
    def test_verify_certificate_checks_bound_consistency(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        bound_not_satisfied = BoundDerivation("PRIVACY_LOSS", (0, 1), 2.0, 1.0, False)
        
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[bound_not_satisfied],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        with pytest.raises(VerificationError, match="claims validity"):
            verify_certificate(cert)


class TestCertificateSerialization:
    """Test certificate save/load."""
    
    def test_save_and_load_certificate(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cert.json"
            save_certificate(cert, path)
            
            loaded = load_certificate(path)
            
            assert loaded.certificate_id == cert.certificate_id
            assert loaded.epsilon == cert.epsilon
            assert loaded.is_valid == cert.is_valid
    
    def test_save_creates_parent_directories(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "cert.json"
            save_certificate(cert, path)
            
            assert path.exists()


class TestCertificateComposition:
    """Test certificate composition."""
    
    def test_compose_certificates_basic(self):
        spec1 = MechanismSpec(2, 2, "hash1", [(0, 1)])
        cert1 = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec1,
            epsilon=0.5,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        spec2 = MechanismSpec(2, 2, "hash2", [(0, 1)])
        cert2 = ProofCertificate(
            certificate_id="cert2",
            mechanism_spec=spec2,
            epsilon=0.5,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        composed = compose_certificates([cert1, cert2], composition_rule="basic")
        
        assert composed.epsilon == 1.0
        assert composed.delta == 0.0
        assert len(composed.parent_certificates) == 2
    
    def test_compose_certificates_rejects_invalid(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        invalid_cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=False,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        with pytest.raises(VerificationError, match="invalid certificates"):
            compose_certificates([invalid_cert])


class TestCertificateChain:
    """Test CertificateChain class."""
    
    def test_certificate_chain_initialization(self):
        chain = CertificateChain()
        
        assert len(chain.certificates) == 0
        assert len(chain.composition_log) == 0
    
    def test_certificate_chain_add_certificate(self):
        chain = CertificateChain()
        
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        chain.add_certificate(cert)
        
        assert len(chain.certificates) == 1
        assert len(chain.composition_log) == 1
    
    def test_certificate_chain_rejects_invalid(self):
        chain = CertificateChain()
        
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        invalid_cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=False,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        with pytest.raises(VerificationError):
            chain.add_certificate(invalid_cert)
    
    def test_certificate_chain_compute_total_privacy(self):
        chain = CertificateChain()
        
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        
        for i in range(3):
            cert = ProofCertificate(
                certificate_id=f"cert{i}",
                mechanism_spec=spec,
                epsilon=0.5,
                delta=0.01,
                is_valid=True,
                verification_method="FLOAT_ARITHMETIC",
                bound_derivations=[],
                tolerance=1e-9,
                confidence=1.0,
                created_at="2024-01-01T00:00:00Z",
                verifier_version="1.0.0",
            )
            chain.add_certificate(cert)
        
        total_eps, total_delta = chain.compute_total_privacy("basic")
        
        assert total_eps == 1.5
        assert total_delta == 0.03
    
    def test_certificate_chain_verify_chain(self):
        chain = CertificateChain()
        
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        bound = BoundDerivation("PRIVACY_LOSS", (0, 1), 0.5, 1.0, True)
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[bound],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        chain.add_certificate(cert)
        
        result = chain.verify_chain()
        assert result is True


class TestCertificateValidator:
    """Test CertificateValidator class."""
    
    def test_validator_initialization(self):
        validator = CertificateValidator()
        
        assert len(validator.validation_rules) > 0
    
    def test_validator_checks_positive_epsilon(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=-1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        validator = CertificateValidator()
        errors = validator.validate(cert)
        
        assert len(errors) > 0
        assert any("epsilon" in e.lower() for e in errors)
    
    def test_validator_checks_nonnegative_delta(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=-0.1,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        validator = CertificateValidator()
        errors = validator.validate(cert)
        
        assert len(errors) > 0
        assert any("delta" in e.lower() for e in errors)
    
    def test_validator_is_valid_method(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        bound = BoundDerivation("PRIVACY_LOSS", (0, 1), 0.5, 1.0, True)
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[bound],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        validator = CertificateValidator()
        
        assert validator.is_valid(cert) is True


class TestBatchOperations:
    """Test batch certificate operations."""
    
    def test_batch_verify_certificates(self):
        spec = MechanismSpec(2, 2, "hash", [(0, 1)])
        bound = BoundDerivation("PRIVACY_LOSS", (0, 1), 0.5, 1.0, True)
        
        certs = []
        for i in range(3):
            cert = ProofCertificate(
                certificate_id=f"cert{i}",
                mechanism_spec=spec,
                epsilon=1.0,
                delta=0.0,
                is_valid=True,
                verification_method="FLOAT_ARITHMETIC",
                bound_derivations=[bound],
                tolerance=1e-9,
                confidence=1.0,
                created_at="2024-01-01T00:00:00Z",
                verifier_version="1.0.0",
            )
            certs.append(cert)
        
        result = batch_verify_certificates(certs)
        
        assert "valid" in result
        assert "invalid" in result
        assert len(result["valid"]) == 3
    
    def test_merge_certificates(self):
        spec1 = MechanismSpec(2, 2, "hash1", [(0, 1)])
        cert1 = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec1,
            epsilon=1.0,
            delta=0.1,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        spec2 = MechanismSpec(2, 2, "hash2", [(0, 1)])
        cert2 = ProofCertificate(
            certificate_id="cert2",
            mechanism_spec=spec2,
            epsilon=0.5,
            delta=0.05,
            is_valid=True,
            verification_method="FLOAT_ARITHMETIC",
            bound_derivations=[],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        merged = merge_certificates(cert1, cert2)
        
        assert merged.epsilon == 1.0
        assert abs(merged.delta - 0.15) < 1e-9
        assert len(merged.parent_certificates) == 2


class TestReportGeneration:
    """Test certificate report generation."""
    
    def test_generate_certificate_report(self):
        spec = MechanismSpec(2, 3, "hash", [(0, 1)])
        bound = BoundDerivation("PRIVACY_LOSS", (0, 1), 0.5, 1.0, True)
        event = AuditEvent("2024-01-01T00:00:00Z", "VERIFICATION", "Started")
        
        cert = ProofCertificate(
            certificate_id="cert123",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="INTERVAL_ARITHMETIC",
            bound_derivations=[bound],
            tolerance=1e-9,
            confidence=0.95,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
            audit_log=[event],
        )
        
        report = generate_certificate_report(cert)
        
        assert "DIFFERENTIAL PRIVACY" in report
        assert "cert123" in report
        assert "BOUND DERIVATIONS" in report
        assert "AUDIT LOG" in report
    
    def test_export_certificate_to_latex(self):
        spec = MechanismSpec(2, 2, "abc123", [(0, 1)])
        bound = BoundDerivation("PRIVACY_LOSS", (0, 1), 0.5, 1.0, True)
        cert = ProofCertificate(
            certificate_id="cert1",
            mechanism_spec=spec,
            epsilon=1.0,
            delta=0.0,
            is_valid=True,
            verification_method="INTERVAL_ARITHMETIC",
            bound_derivations=[bound],
            tolerance=1e-9,
            confidence=1.0,
            created_at="2024-01-01T00:00:00Z",
            verifier_version="1.0.0",
        )
        
        latex = export_certificate_to_latex(cert)
        
        assert "\\documentclass{article}" in latex
        assert "\\epsilon = 1.0" in latex
        assert "\\texttt{abc123}" in latex
        assert "\\end{document}" in latex
