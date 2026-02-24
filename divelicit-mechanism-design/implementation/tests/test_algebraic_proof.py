"""Tests for the algebraic proof of the Sinkhorn-VCG composition theorem."""

import numpy as np
import pytest

from src.algebraic_proof import (
    verify_algebraic_proof,
    verify_exponential_structure,
    verify_payment_independence,
    full_algebraic_verification,
    AlgebraicProofResult,
)


@pytest.fixture
def small_instance():
    """Small instance for testing."""
    rng = np.random.RandomState(42)
    n, d = 10, 8
    embs = rng.randn(n, d)
    quals = rng.uniform(0.3, 0.9, n)
    selected = [0, 2, 4, 6]
    return embs, quals, selected


@pytest.fixture
def medium_instance():
    """Medium instance for testing."""
    rng = np.random.RandomState(123)
    n, d = 20, 16
    embs = rng.randn(n, d)
    quals = rng.uniform(0.2, 0.95, n)
    selected = [1, 3, 5, 7, 9]
    return embs, quals, selected


class TestAlgebraicProof:
    """Test the main algebraic proof of quasi-linearity."""

    def test_proof_passes_basic(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_algebraic_proof(
            embs, quals, selected, n_perturbations=20, seed=42
        )
        assert isinstance(result, AlgebraicProofResult)
        assert result.proof_verified is True
        assert result.div_independent_of_q is True
        assert result.quasi_linear_exact is True

    def test_div_independent_of_quality(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_algebraic_proof(
            embs, quals, selected, n_perturbations=50, seed=42
        )
        assert result.max_div_perturbation < 1e-10

    def test_decomposition_exact(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_algebraic_proof(
            embs, quals, selected, n_perturbations=50, seed=42
        )
        assert result.max_decomposition_error < 1e-8

    def test_potentials_independent(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_algebraic_proof(
            embs, quals, selected, n_perturbations=30, seed=42
        )
        assert result.potentials_independent_of_q is True
        assert result.max_potential_perturbation < 1e-8

    def test_proof_medium_instance(self, medium_instance):
        embs, quals, selected = medium_instance
        result = verify_algebraic_proof(
            embs, quals, selected, n_perturbations=30, seed=42
        )
        assert result.proof_verified is True

    def test_proof_with_different_quality_weights(self, small_instance):
        embs, quals, selected = small_instance
        for qw in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = verify_algebraic_proof(
                embs, quals, selected, quality_weight=qw,
                n_perturbations=10, seed=42
            )
            assert result.proof_verified is True, f"Failed for quality_weight={qw}"

    def test_proof_with_different_reg(self, small_instance):
        embs, quals, selected = small_instance
        for reg in [0.01, 0.1, 0.5, 1.0]:
            result = verify_algebraic_proof(
                embs, quals, selected, reg=reg,
                n_perturbations=10, seed=42
            )
            assert result.proof_verified is True, f"Failed for reg={reg}"

    def test_proof_explanation_nonempty(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_algebraic_proof(
            embs, quals, selected, n_perturbations=10, seed=42
        )
        assert len(result.explanation) > 100

    def test_empty_selection(self):
        embs = np.random.randn(5, 4)
        quals = np.random.uniform(0, 1, 5)
        result = verify_algebraic_proof(embs, quals, [], n_perturbations=5)
        assert result.proof_verified is True


class TestExponentialStructure:
    """Test the exponential structure of Sinkhorn dual potentials."""

    def test_marginals_satisfied(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_exponential_structure(embs, selected, reg=0.1)
        assert result["marginals_satisfied"] is True

    def test_plan_reconstructed(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_exponential_structure(embs, selected, reg=0.1)
        assert result["plan_reconstructed"] is True
        assert result["plan_reconstruction_error"] < 1e-8

    def test_complementary_slackness(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_exponential_structure(embs, selected, reg=0.1)
        assert result["complementary_slackness"] is True

    def test_different_regularizations(self, small_instance):
        embs, quals, selected = small_instance
        for reg in [0.05, 0.1, 0.5]:
            result = verify_exponential_structure(embs, selected, reg=reg)
            assert result["marginals_satisfied"] is True


class TestPaymentIndependence:
    """Test that VCG payments are independent of the paying agent's report."""

    def test_payment_independent(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_payment_independence(
            embs, quals, selected, n_tests=20, seed=42
        )
        assert result["payment_independent"] is True

    def test_max_payment_error_small(self, small_instance):
        embs, quals, selected = small_instance
        result = verify_payment_independence(
            embs, quals, selected, n_tests=20, seed=42
        )
        assert result["max_payment_error"] < 1e-6


class TestFullVerification:
    """Test the full algebraic verification suite."""

    def test_full_verification_passes(self, small_instance):
        embs, quals, selected = small_instance
        result = full_algebraic_verification(
            embs, quals, selected, seed=42
        )
        assert result["all_verified"] is True
        assert result["quasi_linearity"]["verified"] is True
        assert result["exponential_structure"]["complementary_slackness"] is True
        assert result["payment_independence"]["verified"] is True

    def test_full_verification_has_proof_text(self, small_instance):
        embs, quals, selected = small_instance
        result = full_algebraic_verification(
            embs, quals, selected, seed=42
        )
        assert "ALGEBRAIC PROOF" in result["proof_text"]
        assert "PASS" in result["proof_text"]
